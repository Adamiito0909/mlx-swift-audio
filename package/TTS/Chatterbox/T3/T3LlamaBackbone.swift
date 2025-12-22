// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

//  LLaMA backbone for T3 model

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - LLaMA Attention

class T3LlamaAttention: Module {
  let config: T3LlamaConfig
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  let rope: Llama3RoPE

  init(_ config: T3LlamaConfig) {
    self.config = config

    let dim = config.hiddenSize
    let heads = config.numAttentionHeads
    let kvHeads = config.numKeyValueHeads
    let headDim = config.headDim

    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
    _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
    _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
    _oProj.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)

    rope = Llama3RoPE(
      dims: headDim,
      traditional: false,
      base: config.ropeTheta,
      scaleFactor: config.ropeScaling.factor,
      lowFreqFactor: config.ropeScaling.lowFreqFactor,
      highFreqFactor: config.ropeScaling.highFreqFactor,
      oldContextLen: Float(config.ropeScaling.originalMaxPositionEmbeddings),
    )
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?,
  ) -> MLXArray {
    let B = x.shape[0]
    let L = x.shape[1]

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = queries.reshaped([B, L, config.numAttentionHeads, -1]).transposed(0, 2, 1, 3)
    keys = keys.reshaped([B, L, config.numKeyValueHeads, -1]).transposed(0, 2, 1, 3)
    values = values.reshaped([B, L, config.numKeyValueHeads, -1]).transposed(0, 2, 1, 3)

    let offset = cache?.offset ?? 0
    queries = rope(queries, offset: offset)
    keys = rope(keys, offset: offset)

    // Use attentionWithCacheUpdate for automatic routing to quantized or regular attention
    let attnResult = attentionWithCacheUpdate(
      queries: queries,
      keys: keys,
      values: values,
      cache: cache,
      scale: scale,
      mask: mask,
    )

    let output = attnResult.transposed(0, 2, 1, 3).reshaped([B, L, config.numAttentionHeads * config.headDim])
    return oProj(output)
  }
}

// MARK: - Transformer Block

class T3TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: T3LlamaAttention
  @ModuleInfo var mlp: SwiGLUMLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(_ config: T3LlamaConfig) {
    _attention.wrappedValue = T3LlamaAttention(config)
    _mlp.wrappedValue = SwiGLUMLP(
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      bias: config.mlpBias,
    )
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    cache: KVCache?,
  ) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    return h + r
  }
}

// MARK: - LLaMA Model Inner

/// Inner model class for weight path matching (tfmr.model.layers, tfmr.model.norm)
class T3LlamaModel: Module {
  @ModuleInfo var layers: [T3TransformerBlock]
  @ModuleInfo var norm: RMSNorm

  init(_ config: T3LlamaConfig) {
    _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in T3TransformerBlock(config) }
    _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
  }
}

// MARK: - LLaMA Backbone

/// LLaMA backbone for T3 model
class T3LlamaBackbone: Module {
  let config: T3LlamaConfig
  let kvHeads: [Int]

  @ModuleInfo var model: T3LlamaModel

  init(_ config: T3LlamaConfig) {
    self.config = config
    kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    _model.wrappedValue = T3LlamaModel(config)
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = inputs

    let mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal

    for (i, layer) in model.layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return model.norm(h)
  }

  /// Create KV caches for all layers
  /// - Parameters:
  ///   - quantized: If true, use QuantizedKVCache for reduced memory. Note: quantization adds
  ///     overhead that may exceed benefits for typical sequence lengths. Only enable for very
  ///     long sequences where memory is constrained.
  ///   - groupSize: Group size for quantization (default 64)
  ///   - bits: Number of bits for quantization (default 4)
  func newCache(
    quantized: Bool = false,
    groupSize: Int = 64,
    bits: Int = 4,
  ) -> [KVCache] {
    if quantized {
      (0 ..< config.numHiddenLayers).map { _ in
        QuantizedKVCache(groupSize: groupSize, bits: bits)
      }
    } else {
      (0 ..< config.numHiddenLayers).map { _ in KVCacheSimple() }
    }
  }
}
