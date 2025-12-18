// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXNN

/// Single SANM encoder layer
///
/// Structure (pre-norm):
/// - LayerNorm -> Self-Attention (SANM) -> Dropout -> Residual
/// - LayerNorm -> Feed-Forward -> Dropout -> Residual
///
/// Matches the original FunASR implementation.
class EncoderLayerSANM: Module {
  let inSize: Int
  let size: Int

  @ModuleInfo var norm1: LayerNorm
  @ModuleInfo(key: "self_attn") var selfAttn: MultiHeadAttentionSANM
  @ModuleInfo var norm2: LayerNorm
  @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward
  @ModuleInfo var dropout: Dropout

  /// Initialize a SANM encoder layer
  ///
  /// - Parameters:
  ///   - inSize: Input dimension
  ///   - size: Output dimension
  ///   - nHead: Number of attention heads
  ///   - dFF: Feed-forward hidden dimension
  ///   - kernelSize: FSMN kernel size (default: 11)
  ///   - sanmShift: SANM shift for asymmetric context (default: 0)
  ///   - dropoutRate: Dropout rate (default: 0.0)
  init(
    inSize: Int,
    size: Int,
    nHead: Int,
    dFF: Int,
    kernelSize: Int = 11,
    sanmShift: Int = 0,
    dropoutRate: Float = 0.0
  ) {
    self.inSize = inSize
    self.size = size

    _norm1.wrappedValue = LayerNorm(dimensions: inSize)
    _selfAttn.wrappedValue = MultiHeadAttentionSANM(
      nHead: nHead,
      inFeat: inSize,
      nFeat: size,
      kernelSize: kernelSize,
      sanmShift: sanmShift,
      dropoutRate: dropoutRate
    )

    _norm2.wrappedValue = LayerNorm(dimensions: size)
    _feedForward.wrappedValue = PositionwiseFeedForward(
      idim: size,
      hiddenUnits: dFF,
      dropoutRate: dropoutRate
    )

    _dropout.wrappedValue = Dropout(p: dropoutRate)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, seq, inSize)
  ///   - mask: Optional attention mask
  /// - Returns: Output tensor (batch, seq, size)
  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    // Self-attention with pre-norm
    let residual = x
    var out = norm1(x)
    out = selfAttn(out, mask: mask)
    out = dropout(out)

    // Add residual (only if dimensions match)
    if inSize == size {
      out = out + residual
    }

    // Feed-forward with pre-norm
    let residual2 = out
    out = norm2(out)
    out = feedForward(out)
    out = out + residual2

    return out
  }
}

/// Standard encoder layer (without FSMN)
///
/// Used in the AudioAdaptor's transformer blocks.
/// Structure (pre-norm, matches original FunASR):
/// - LayerNorm -> Self-Attention -> Dropout -> Residual
/// - LayerNorm -> Feed-Forward -> Dropout -> Residual
class FunASREncoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttn: FunASRMultiHeadAttention
  @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward
  @ModuleInfo var norm1: LayerNorm
  @ModuleInfo var norm2: LayerNorm
  @ModuleInfo var dropout: Dropout

  /// Initialize a standard encoder layer
  ///
  /// - Parameters:
  ///   - size: Feature dimension
  ///   - nHead: Number of attention heads
  ///   - dFF: Feed-forward hidden dimension
  ///   - dropoutRate: Dropout rate (default: 0.0)
  init(
    size: Int,
    nHead: Int,
    dFF: Int,
    dropoutRate: Float = 0.0
  ) {
    _selfAttn.wrappedValue = FunASRMultiHeadAttention(
      nHead: nHead,
      nFeat: size,
      dropoutRate: dropoutRate
    )
    _feedForward.wrappedValue = PositionwiseFeedForward(
      idim: size,
      hiddenUnits: dFF,
      dropoutRate: dropoutRate
    )
    _norm1.wrappedValue = LayerNorm(dimensions: size)
    _norm2.wrappedValue = LayerNorm(dimensions: size)
    _dropout.wrappedValue = Dropout(p: dropoutRate)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, seq, size)
  ///   - mask: Optional attention mask
  /// - Returns: Output tensor (batch, seq, size)
  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    // Self-attention with pre-norm (matches original FunASR)
    var residual = x
    var out = norm1(x)
    out = selfAttn(query: out, key: out, value: out, mask: mask)
    out = residual + dropout(out)

    // Feed-forward with pre-norm (matches original FunASR)
    residual = out
    out = norm2(out)
    out = feedForward(out)
    out = residual + dropout(out)

    return out
  }
}
