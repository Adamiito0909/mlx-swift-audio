// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Qwen3 Attention

/// Qwen3 attention with Grouped Query Attention (GQA) and RoPE
///
/// Key features:
/// - GQA: 16 query heads, 8 KV heads
/// - QK normalization with per-head RMSNorm (Qwen3 specific)
/// - Standard RoPE with base=1000000
/// - KV cache support for efficient inference
class Qwen3Attention: Module {
  let config: Qwen3Config
  let nHeads: Int
  let nKVHeads: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  // QK normalization (per-head RMSNorm) - Qwen3 specific
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  // Rotary embeddings
  let rope: RoPE

  /// Initialize Qwen3 attention
  ///
  /// - Parameter config: Qwen3 configuration
  init(config: Qwen3Config) {
    self.config = config
    nHeads = config.numAttentionHeads
    nKVHeads = config.numKeyValueHeads
    headDim = config.headDim
    scale = pow(Float(headDim), -0.5)

    let dim = config.hiddenSize

    // Projections (no bias in Qwen3)
    _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
    _kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
    _vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
    _oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

    // QK normalization
    _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

    // Rotary embeddings
    rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, seq, hiddenSize)
  ///   - mask: Optional attention mask
  ///   - cache: Optional KV cache (keys, values)
  /// - Returns: Tuple of (output, new cache)
  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
    cache: (MLXArray, MLXArray)? = nil
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    // Reshape for multi-head attention
    queries = queries.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
    keys = keys.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
    values = values.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

    // Apply QK normalization
    queries = qNorm(queries)
    keys = kNorm(keys)

    // Apply RoPE
    if let cache {
      let offset = cache.0.shape[2]
      queries = rope(queries, offset: offset)
      keys = rope(keys, offset: offset)
      keys = MLX.concatenated([cache.0, keys], axis: 2)
      values = MLX.concatenated([cache.1, values], axis: 2)
    } else {
      queries = rope(queries)
      keys = rope(keys)
    }

    let newCache = (keys, values)

    // Scaled dot-product attention
    let output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: mask
    )

    // Reshape and project
    let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])

    return (oProj(outputReshaped), newCache)
  }
}

// MARK: - Qwen3 MLP

/// Qwen3 MLP with SwiGLU activation
///
/// Structure: down_proj(silu(gate_proj(x)) * up_proj(x))
class Qwen3MLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  /// Initialize Qwen3 MLP
  ///
  /// - Parameter config: Qwen3 configuration
  init(config: Qwen3Config) {
    let dim = config.hiddenSize
    let hiddenDim = config.intermediateSize

    _gateProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
    _upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
    _downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
  }

  /// Forward pass
  ///
  /// - Parameter x: Input tensor (batch, seq, hiddenSize)
  /// - Returns: Output tensor (batch, seq, hiddenSize)
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}

// MARK: - Qwen3 Transformer Block

/// Single Qwen3 transformer block
///
/// Structure (pre-norm):
/// - RMSNorm -> Self-Attention -> Residual
/// - RMSNorm -> MLP -> Residual
class Qwen3TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
  @ModuleInfo var mlp: Qwen3MLP
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  /// Initialize a Qwen3 transformer block
  ///
  /// - Parameter config: Qwen3 configuration
  init(config: Qwen3Config) {
    _selfAttn.wrappedValue = Qwen3Attention(config: config)
    _mlp.wrappedValue = Qwen3MLP(config: config)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, seq, hiddenSize)
  ///   - mask: Optional attention mask
  ///   - cache: Optional KV cache
  /// - Returns: Tuple of (output, new cache)
  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray? = nil,
    cache: (MLXArray, MLXArray)? = nil
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    // Self-attention with pre-norm and residual
    var r = x
    var h = inputLayerNorm(x)
    let (attnOut, newCache) = selfAttn(h, mask: mask, cache: cache)
    h = r + attnOut

    // MLP with pre-norm and residual
    r = h
    h = postAttentionLayerNorm(h)
    h = mlp(h)
    h = r + h

    return (h, newCache)
  }
}

// MARK: - Qwen3 Model

/// Qwen3 transformer model (without LM head)
class Qwen3Model: Module {
  let config: Qwen3Config

  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo var layers: [Qwen3TransformerBlock]
  @ModuleInfo var norm: RMSNorm

  /// Initialize Qwen3 model
  ///
  /// - Parameter config: Qwen3 configuration
  init(config: Qwen3Config) {
    self.config = config

    _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
    _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
      Qwen3TransformerBlock(config: config)
    }
    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs (batch, seq) - optional if inputEmbeddings provided
  ///   - inputEmbeddings: Pre-computed embeddings (batch, seq, hiddenSize) - optional
  ///   - mask: Optional attention mask
  ///   - cache: Optional KV cache for all layers
  /// - Returns: Tuple of (hidden states, new cache)
  func callAsFunction(
    inputIds: MLXArray? = nil,
    inputEmbeddings: MLXArray? = nil,
    mask: MLXArray? = nil,
    cache: [(MLXArray, MLXArray)?]? = nil
  ) -> (MLXArray, [(MLXArray, MLXArray)?]) {
    var h: MLXArray
    if let embeddings = inputEmbeddings {
      h = embeddings
    } else if let ids = inputIds {
      h = embedTokens(ids)
    } else {
      fatalError("Either inputIds or inputEmbeddings must be provided")
    }

    // Create causal mask if needed
    var actualMask = mask
    if mask == nil, h.shape[1] > 1 {
      let seqLen = h.shape[1]
      let indices = MLXArray(0 ..< seqLen)
      // Create causal mask: position i can attend to positions 0..i
      let causalMask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
      actualMask = MLX.where(causalMask, MLXArray(-Float.infinity), MLXArray(0.0))
      actualMask = actualMask?.asType(h.dtype)
    }

    // Initialize cache if not provided
    var actualCache: [(MLXArray, MLXArray)?] = cache ?? Array(repeating: nil, count: layers.count)

    // Process through layers
    var newCache: [(MLXArray, MLXArray)?] = []
    for (i, layer) in layers.enumerated() {
      let (layerOut, layerCache) = layer(h, mask: actualMask, cache: actualCache[i])
      h = layerOut
      newCache.append(layerCache)
    }

    return (norm(h), newCache)
  }
}

// MARK: - Qwen3 For Causal LM

/// Qwen3 model with language modeling head
class Qwen3ForCausalLM: Module {
  let config: Qwen3Config
  @ModuleInfo var model: Qwen3Model

  // LM head (only if not using tied embeddings)
  @ModuleInfo(key: "lm_head") var lmHead: Linear?

  /// Initialize Qwen3 for causal LM
  ///
  /// - Parameter config: Qwen3 configuration
  init(config: Qwen3Config) {
    self.config = config
    _model.wrappedValue = Qwen3Model(config: config)

    if !config.tieWordEmbeddings {
      _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    } else {
      _lmHead.wrappedValue = nil
    }
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs (batch, seq) - optional if inputEmbeddings provided
  ///   - inputEmbeddings: Pre-computed embeddings - optional
  ///   - mask: Optional attention mask
  ///   - cache: Optional KV cache
  /// - Returns: Tuple of (logits, new cache)
  func callAsFunction(
    inputIds: MLXArray? = nil,
    inputEmbeddings: MLXArray? = nil,
    mask: MLXArray? = nil,
    cache: [(MLXArray, MLXArray)?]? = nil
  ) -> (MLXArray, [(MLXArray, MLXArray)?]) {
    let (out, newCache) = model(
      inputIds: inputIds,
      inputEmbeddings: inputEmbeddings,
      mask: mask,
      cache: cache
    )

    let logits: MLXArray
    if config.tieWordEmbeddings {
      logits = model.embedTokens.asLinear(out)
    } else if let lmHead {
      logits = lmHead(out)
    } else {
      fatalError("LM head not initialized and embeddings not tied")
    }

    return (logits, newCache)
  }

  /// Get the input embedding layer
  ///
  /// - Returns: The embedding layer for getting token embeddings
  func getInputEmbeddings() -> Embedding {
    model.embedTokens
  }
}
