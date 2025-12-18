// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

import Foundation
import MLX
import MLXNN

/// Text decoder for Whisper
///
/// Autoregressive transformer decoder with:
/// - Token and positional embeddings
/// - Self-attention with causal mask
/// - Cross-attention to audio encoder output
/// - Output projection to vocabulary
class TextDecoder: Module {
  @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
  @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
  @ModuleInfo var blocks: [ResidualAttentionBlock]
  @ModuleInfo var ln: LayerNorm
  let mask: MLXArray

  init(nVocab: Int, nCtx: Int, nState: Int, nHead: Int, nLayer: Int) {
    _tokenEmbedding.wrappedValue = Embedding(embeddingCount: nVocab, dimensions: nState)

    // Learned positional embeddings (initialized to zeros, loaded from checkpoint)
    _positionalEmbedding.wrappedValue = MLXArray.zeros([nCtx, nState])

    // Transformer blocks with cross-attention
    _blocks.wrappedValue = (0 ..< nLayer).map { _ in
      ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)
    }

    _ln.wrappedValue = LayerNorm(dimensions: nState)

    // Causal mask for autoregressive decoding
    // Create additive causal mask using -inf (matches Python's create_additive_causal_mask)
    // Using -inf works correctly in any dtype and ensures proper masking in softmax
    let indices = MLXArray(0 ..< nCtx)
    let causalMask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
    mask = MLX.where(causalMask, -Float.infinity, MLXArray(Float(0)))
    // Note: mask stays as float32 here; MLX will handle dtype promotion during addition
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Token indices (batch, n_tokens)
  ///   - xa: Encoded audio features (batch, n_audio_ctx, n_audio_state)
  ///   - kvCache: Optional cached key/value tensors from previous steps
  /// - Returns: Tuple of (logits, new_kv_cache, cross_attention_weights)
  func callAsFunction(
    _ x: MLXArray,
    xa: MLXArray,
    kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
  ) -> (MLXArray, [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)], [MLXArray?]) {
    // Determine offset for positional embeddings (for KV caching)
    let offset: Int = if let kvCache, let firstSelfKv = kvCache.first?.0 {
      firstSelfKv.0.shape[1] // Use cached sequence length
    } else {
      0
    }

    // Token embeddings + positional embeddings
    let nTokens = x.shape[x.ndim - 1]
    var output = tokenEmbedding(x) + positionalEmbedding[offset ..< (offset + nTokens)]

    // Initialize KV cache if not provided
    var newKvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)] = []
    var crossQk: [MLXArray?] = []

    let kvCacheToUse = kvCache ?? Array(repeating: (nil, nil), count: blocks.count)

    // Apply transformer blocks
    for (i, block) in blocks.enumerated() {
      let (blockOutput, blockKvCache, blockCrossQk) = block(
        output,
        xa: xa,
        mask: mask,
        kvCache: kvCacheToUse[i],
        offset: offset
      )
      output = blockOutput
      newKvCache.append(blockKvCache)
      crossQk.append(blockCrossQk)
    }

    // Final layer norm
    output = ln(output)

    // Project to vocabulary using token embedding weights (weight tying)
    let logits = tokenEmbedding.asLinear(output)

    return (logits, newKvCache, crossQk)
  }
}
