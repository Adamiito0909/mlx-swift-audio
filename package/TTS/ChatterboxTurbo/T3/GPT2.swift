// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// GPT2 backbone for Chatterbox Turbo T3 model

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - GPT2 Config

/// Configuration for GPT2 model
struct GPT2Config: Codable, Sendable {
  var modelType: String = "gpt2"
  var vocabSize: Int = 50276
  var nPositions: Int = 8196
  var nEmbd: Int = 1024
  var nLayer: Int = 24
  var nHead: Int = 16
  var nInner: Int? = nil // Defaults to 4 * nEmbd
  var activationFunction: String = "gelu_new"
  var residPdrop: Float = 0.1
  var embdPdrop: Float = 0.1
  var attnPdrop: Float = 0.1
  var layerNormEpsilon: Float = 1e-5

  var hiddenSize: Int { nEmbd }
  var numAttentionHeads: Int { nHead }
  var numHiddenLayers: Int { nLayer }
  var intermediateSize: Int { nInner ?? (4 * nEmbd) }
  var headDim: Int { nEmbd / nHead }

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case vocabSize = "vocab_size"
    case nPositions = "n_positions"
    case nEmbd = "n_embd"
    case nLayer = "n_layer"
    case nHead = "n_head"
    case nInner = "n_inner"
    case activationFunction = "activation_function"
    case residPdrop = "resid_pdrop"
    case embdPdrop = "embd_pdrop"
    case attnPdrop = "attn_pdrop"
    case layerNormEpsilon = "layer_norm_epsilon"
  }

  init() {}

  /// Create GPT2 Medium configuration for T3 Turbo
  static func gpt2Medium() -> GPT2Config {
    GPT2Config()
  }
}

// MARK: - GELU New Activation

/// GELU activation function (OpenAI GPT variant)
/// Implementation matches HuggingFace transformers gelu_new
func geluNew(_ x: MLXArray) -> MLXArray {
  0.5 * x * (1.0 + MLX.tanh(sqrt(2.0 / Float.pi) * (x + 0.044715 * MLX.pow(x, 3))))
}

// MARK: - GPT2 Attention

/// GPT2 multi-head attention with combined QKV projection
class GPT2Attention: Module {
  let config: GPT2Config
  let scale: Float

  @ModuleInfo(key: "c_attn") var cAttn: Linear
  @ModuleInfo(key: "c_proj") var cProj: Linear

  init(_ config: GPT2Config) {
    self.config = config
    scale = pow(Float(config.headDim), -0.5)

    // Combined QKV projection (3 * hidden_size)
    _cAttn.wrappedValue = Linear(config.nEmbd, 3 * config.nEmbd)
    _cProj.wrappedValue = Linear(config.nEmbd, config.nEmbd)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask _: MLXArray? = nil,
    cache: KVCache?
  ) -> MLXArray {
    let B = hiddenStates.shape[0]
    let T = hiddenStates.shape[1]

    // Combined QKV projection
    let qkv = cAttn(hiddenStates)
    let split = qkv.split(parts: 3, axis: -1)
    var q = split[0]
    var k = split[1]
    var v = split[2]

    // Reshape to (B, numHeads, T, headDim)
    q = q.reshaped([B, T, config.nHead, config.headDim]).transposed(0, 2, 1, 3)
    k = k.reshaped([B, T, config.nHead, config.headDim]).transposed(0, 2, 1, 3)
    v = v.reshaped([B, T, config.nHead, config.headDim]).transposed(0, 2, 1, 3)

    // Update KV cache if present
    if let cache {
      (k, v) = cache.update(keys: k, values: v)
    }

    // Use optimized scaled dot product attention
    // .causal for prefill (T > 1), .none for single-token generation
    let L = q.shape[2]
    let attnOutput = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: L > 1 ? .causal : .none
    ).transposed(0, 2, 1, 3).reshaped([B, T, config.nEmbd])

    // Output projection
    return cProj(attnOutput)
  }
}

// MARK: - GPT2 MLP

/// GPT2 feed-forward MLP with gelu_new activation
class GPT2MLP: Module {
  @ModuleInfo(key: "c_fc") var cFc: Linear
  @ModuleInfo(key: "c_proj") var cProj: Linear

  init(_ config: GPT2Config) {
    let innerDim = config.intermediateSize
    _cFc.wrappedValue = Linear(config.nEmbd, innerDim)
    _cProj.wrappedValue = Linear(innerDim, config.nEmbd)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = cFc(x)
    h = geluNew(h)
    return cProj(h)
  }
}

// MARK: - GPT2 Block

/// GPT2 transformer block (Pre-LN architecture)
class GPT2Block: Module {
  @ModuleInfo(key: "ln_1") var ln1: LayerNorm
  @ModuleInfo var attn: GPT2Attention
  @ModuleInfo(key: "ln_2") var ln2: LayerNorm
  @ModuleInfo var mlp: GPT2MLP

  init(_ config: GPT2Config) {
    _ln1.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
    _attn.wrappedValue = GPT2Attention(config)
    _ln2.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
    _mlp.wrappedValue = GPT2MLP(config)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
    cache: KVCache?
  ) -> MLXArray {
    // Self-attention with residual
    let residual = hiddenStates
    var h = ln1(hiddenStates)
    h = attn(h, attentionMask: attentionMask, cache: cache)
    h = residual + h

    // MLP with residual
    let residual2 = h
    h = ln2(h)
    h = mlp(h)
    return residual2 + h
  }
}

// MARK: - GPT2 Model

/// GPT2 base model (without LM head)
class GPT2Model: Module {
  let config: GPT2Config

  @ModuleInfo(key: "wte") var wte: Embedding
  @ModuleInfo(key: "wpe") var wpe: Embedding
  @ModuleInfo var h: [GPT2Block]
  @ModuleInfo(key: "ln_f") var lnF: LayerNorm

  init(_ config: GPT2Config) {
    self.config = config

    // Token and position embeddings
    _wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
    _wpe.wrappedValue = Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbd)

    // Transformer blocks
    _h.wrappedValue = (0 ..< config.nLayer).map { _ in GPT2Block(config) }

    // Final layer norm
    _lnF.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
  }

  /// Forward pass
  /// - Parameters:
  ///   - inputIds: Token IDs (B, T) - optional if inputsEmbeds provided
  ///   - inputsEmbeds: Pre-computed embeddings (B, T, D) - takes precedence over inputIds
  ///   - attentionMask: Optional attention mask
  ///   - cache: Optional list of KV caches for each layer
  /// - Returns: Hidden states (B, T, D)
  func callAsFunction(
    inputIds: MLXArray? = nil,
    inputsEmbeds: MLXArray? = nil,
    attentionMask: MLXArray? = nil,
    cache: [KVCache]? = nil
  ) -> MLXArray {
    var hiddenStates: MLXArray
    if let embeds = inputsEmbeds {
      hiddenStates = embeds
    } else if let ids = inputIds {
      hiddenStates = wte(ids)
    } else {
      fatalError("Either inputIds or inputsEmbeds must be provided")
    }

    let T = hiddenStates.shape[1]

    // Add positional embeddings
    let pastLength: Int = if let cache, !cache.isEmpty {
      cache[0].offset
    } else {
      0
    }

    let positionIds = MLXArray(Int32(pastLength) ..< Int32(pastLength + T))
    let positionEmbeds = wpe(positionIds)
    hiddenStates = hiddenStates + positionEmbeds

    // Forward through transformer blocks
    for (i, block) in h.enumerated() {
      hiddenStates = block(
        hiddenStates,
        attentionMask: attentionMask,
        cache: cache?[i]
      )
    }

    // Final layer norm
    return lnF(hiddenStates)
  }

  /// Create KV caches for all layers
  func newCache() -> [KVCache] {
    (0 ..< config.nLayer).map { _ in KVCacheSimple() }
  }
}
