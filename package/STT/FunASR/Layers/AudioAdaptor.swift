// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXNN

/// Audio Adaptor that projects encoder output to LLM embedding space
///
/// Architecture (matches original FunASR Transformer adaptor):
/// - Downsample by grouping k consecutive frames
/// - linear1: encoder_dim * k -> ffn_dim (with ReLU)
/// - linear2: ffn_dim -> llm_dim
/// - blocks: n_layer transformer blocks for refinement
///
/// The downsampling reduces sequence length by factor k while
/// increasing feature dimension.
class AudioAdaptor: Module {
  let config: AudioAdaptorConfig
  let k: Int

  // Linear projections (after downsampling, input dim is encoder_dim * k)
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  // Transformer blocks (optional, based on n_layer)
  @ModuleInfo var blocks: [FunASREncoderLayer]?

  /// Initialize the audio adaptor
  ///
  /// - Parameter config: Adaptor configuration
  init(config: AudioAdaptorConfig) {
    self.config = config
    k = config.downsampleRate

    // Linear projections
    _linear1.wrappedValue = Linear(config.encoderDim * k, config.ffnDim)
    _linear2.wrappedValue = Linear(config.ffnDim, config.llmDim)

    // Transformer blocks (optional, based on n_layer)
    if config.nLayer > 0 {
      // FFN dimension in transformer blocks is llm_dim // 4 (matches original)
      let blockFFNDim = config.llmDim / 4
      _blocks.wrappedValue = (0 ..< config.nLayer).map { _ in
        FunASREncoderLayer(
          size: config.llmDim,
          nHead: config.attentionHeads,
          dFF: blockFFNDim,
          dropoutRate: config.dropout
        )
      }
    } else {
      _blocks.wrappedValue = nil
    }
  }

  /// Forward pass through the adaptor
  ///
  /// - Parameters:
  ///   - x: Encoder output (batch, seq, encoderDim)
  ///   - lengths: Optional sequence lengths
  /// - Returns: Tuple of (projected features, output lengths)
  ///   - Projected features: (batch, seq/k, llmDim)
  ///   - Output lengths: Updated for downsampling
  func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let (batchSize, seqLen, dim) = (x.shape[0], x.shape[1], x.shape[2])

    // Pad sequence to be divisible by k
    let chunkNum = (seqLen - 1) / k + 1
    let padNum = chunkNum * k - seqLen

    var padded = x
    if padNum > 0 {
      padded = MLX.padded(x, widths: [
        IntOrPair(integerLiteral: 0),
        IntOrPair((0, padNum)),
        IntOrPair(integerLiteral: 0),
      ])
    }

    // Reshape to group k consecutive frames
    // (batch, seq, dim) -> (batch, seq/k, dim*k)
    var out = padded.reshaped([batchSize, chunkNum, dim * k])

    // Linear projections with ReLU (matches original)
    out = linear1(out)
    out = relu(out)
    out = linear2(out)

    // Compute output lengths
    let outLengths: MLXArray = if let lengths {
      (lengths - 1) / k + 1
    } else {
      MLXArray(Array(repeating: Int32(chunkNum), count: batchSize))
    }

    // Create padding mask for transformer blocks
    var mask: MLXArray? = nil
    if lengths != nil, blocks != nil {
      // Create mask from lengths
      let maxLen = out.shape[1]
      let indices = MLXArray(0 ..< maxLen).expandedDimensions(axis: 0)
      mask = indices .< outLengths.expandedDimensions(axis: 1)
      // Expand for attention: (batch, 1, 1, seq)
      mask = mask?.expandedDimensions(axes: [1, 2])
    }

    // Apply transformer blocks
    if let blocks {
      for block in blocks {
        out = block(out, mask: mask)
      }
    }

    return (out, outLengths)
  }
}
