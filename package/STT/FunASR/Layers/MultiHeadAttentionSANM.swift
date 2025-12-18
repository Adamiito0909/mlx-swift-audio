// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXFast
import MLXNN

/// Self-Attention with Memory (SANM)
///
/// Combines standard multi-head attention with FSMN (Feedforward Sequential Memory Network)
/// for local context modeling. The FSMN output is added AFTER computing attention (as a residual).
///
/// This matches the original FunASR implementation exactly.
class MultiHeadAttentionSANM: Module {
  let dK: Int
  let h: Int
  let nFeat: Int

  @ModuleInfo(key: "linear_q_k_v") var linearQKV: Linear
  @ModuleInfo(key: "linear_out") var linearOut: Linear
  @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d
  @ModuleInfo var dropout: Dropout

  let leftPadding: Int
  let rightPadding: Int
  let kernelSize: Int

  /// Initialize SANM attention
  ///
  /// - Parameters:
  ///   - nHead: Number of attention heads
  ///   - inFeat: Input feature dimension
  ///   - nFeat: Output feature dimension
  ///   - kernelSize: FSMN kernel size (default: 11)
  ///   - sanmShift: SANM shift for asymmetric context (default: 0)
  ///   - dropoutRate: Dropout rate (default: 0.0)
  init(
    nHead: Int,
    inFeat: Int,
    nFeat: Int,
    kernelSize: Int = 11,
    sanmShift: Int = 0,
    dropoutRate: Float = 0.0
  ) {
    precondition(nFeat % nHead == 0, "nFeat must be divisible by nHead")

    dK = nFeat / nHead
    h = nHead
    self.nFeat = nFeat
    self.kernelSize = kernelSize

    // Combined Q/K/V projection
    _linearQKV.wrappedValue = Linear(inFeat, nFeat * 3)

    // Output projection
    _linearOut.wrappedValue = Linear(nFeat, nFeat)

    // FSMN block - depthwise conv (groups=nFeat) with no padding in conv itself
    // Padding is applied explicitly before conv
    _fsmnBlock.wrappedValue = Conv1d(
      inputChannels: nFeat,
      outputChannels: nFeat,
      kernelSize: kernelSize,
      stride: 1,
      padding: 0,
      groups: nFeat,
      bias: false
    )

    // Compute padding amounts
    var leftPad = (kernelSize - 1) / 2
    if sanmShift > 0 {
      leftPad = leftPad + sanmShift
    }
    let rightPad = kernelSize - 1 - leftPad
    leftPadding = leftPad
    rightPadding = rightPad

    _dropout.wrappedValue = Dropout(p: dropoutRate)
  }

  /// Apply FSMN to inputs
  ///
  /// - Parameters:
  ///   - inputs: The unprojected value tensor (batch, seq, dim)
  ///   - mask: Optional mask tensor
  /// - Returns: FSMN output with local context (batch, seq, dim)
  private func forwardFSMN(_ inputs: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    // Reshape mask once for reuse (broadcast to [batch, seq, 1])
    let maskReshaped: MLXArray? = mask.map { m in
      switch m.ndim {
        case 3: m.reshaped([inputs.shape[0], -1, 1])
        case 2: m.expandedDimensions(axis: -1)
        default: m
      }
    }

    // Apply mask before conv
    var x = maskReshaped.map { inputs * $0 } ?? inputs

    // Apply explicit padding and depthwise conv
    if leftPadding > 0 || rightPadding > 0 {
      x = MLX.padded(x, widths: [
        IntOrPair(integerLiteral: 0),
        IntOrPair((leftPadding, rightPadding)),
        IntOrPair(integerLiteral: 0),
      ])
    }
    x = fsmnBlock(x)

    // Residual connection, dropout, and mask again
    x = dropout(x + inputs)
    return maskReshaped.map { x * $0 } ?? x
  }

  /// Forward pass for SANM attention
  ///
  /// - Parameters:
  ///   - x: Input tensor (batch, seq, inFeat)
  ///   - mask: Optional attention mask
  /// - Returns: Output tensor (batch, seq, nFeat)
  func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let (batchSize, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

    // Combined Q/K/V projection and split
    let qkv = linearQKV(x)
    let qkvSplit = qkv.split(parts: 3, axis: -1)
    let (q, k, v) = (qkvSplit[0], qkvSplit[1], qkvSplit[2])

    // Apply FSMN to unprojected value (before multi-head reshape)
    let fsmnMemory = forwardFSMN(v, mask: mask)

    // Reshape for multi-head attention
    // (batch, seq, nFeat) -> (batch, nHead, seq, dK)
    let qH = q.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)
    let kH = k.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)
    let vH = v.reshaped([batchSize, seqLen, h, dK]).transposed(0, 2, 1, 3)

    // Convert mask to additive format for fast attention if provided
    var attnMask: MLXArray? = nil
    if let mask {
      let expandedMask: MLXArray = if mask.ndim == 2 {
        mask.expandedDimensions(axes: [1, 2]) // (batch, 1, 1, seq)
      } else if mask.ndim == 3 {
        mask.expandedDimensions(axis: 1) // (batch, 1, seq, seq)
      } else {
        mask
      }
      // Convert boolean/binary mask to additive mask (0 -> -inf for masked positions)
      attnMask = MLX.where(expandedMask .== 0, MLXArray(-Float.infinity), MLXArray(0.0))
    }

    // Use fast scaled dot-product attention
    var context = MLXFast.scaledDotProductAttention(
      queries: qH,
      keys: kH,
      values: vH,
      scale: Float(pow(Double(dK), -0.5)),
      mask: attnMask
    )

    // Apply dropout after attention
    context = dropout(context)

    // Reshape back: (batch, nHead, seq, dK) -> (batch, seq, nFeat)
    context = context.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, nFeat])

    // Output projection
    let attOuts = linearOut(context)

    // Add FSMN memory AFTER attention (key difference from naive implementation)
    return attOuts + fsmnMemory
  }
}

/// Standard multi-head attention (without FSMN)
///
/// Used in the AudioAdaptor's transformer blocks
class FunASRMultiHeadAttention: Module {
  let dK: Int
  let h: Int
  let nFeat: Int

  @ModuleInfo(key: "linear_q") var linearQ: Linear
  @ModuleInfo(key: "linear_k") var linearK: Linear
  @ModuleInfo(key: "linear_v") var linearV: Linear
  @ModuleInfo(key: "linear_out") var linearOut: Linear
  @ModuleInfo var dropout: Dropout

  /// Initialize standard multi-head attention
  ///
  /// - Parameters:
  ///   - nHead: Number of attention heads
  ///   - nFeat: Feature dimension
  ///   - dropoutRate: Dropout rate (default: 0.0)
  init(nHead: Int, nFeat: Int, dropoutRate: Float = 0.0) {
    precondition(nFeat % nHead == 0, "nFeat must be divisible by nHead")

    dK = nFeat / nHead
    h = nHead
    self.nFeat = nFeat

    _linearQ.wrappedValue = Linear(nFeat, nFeat)
    _linearK.wrappedValue = Linear(nFeat, nFeat)
    _linearV.wrappedValue = Linear(nFeat, nFeat)
    _linearOut.wrappedValue = Linear(nFeat, nFeat)

    _dropout.wrappedValue = Dropout(p: dropoutRate)
  }

  /// Forward pass
  ///
  /// - Parameters:
  ///   - query: Query tensor (batch, seq, nFeat)
  ///   - key: Key tensor (batch, seq, nFeat)
  ///   - value: Value tensor (batch, seq, nFeat)
  ///   - mask: Optional attention mask
  /// - Returns: Output tensor (batch, seq, nFeat)
  func callAsFunction(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    mask: MLXArray? = nil
  ) -> MLXArray {
    let batchSize = query.shape[0]

    let q = linearQ(query)
    let k = linearK(key)
    let v = linearV(value)

    // Reshape for multi-head attention
    let qH = q.reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)
    let kH = k.reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)
    let vH = v.reshaped([batchSize, -1, h, dK]).transposed(0, 2, 1, 3)

    // Convert mask to additive format
    var attnMask: MLXArray? = nil
    if let mask {
      attnMask = MLX.where(mask .== 0, MLXArray(-Float.infinity), MLXArray(0.0))
    }

    // Use fast scaled dot-product attention
    var context = MLXFast.scaledDotProductAttention(
      queries: qH,
      keys: kH,
      values: vH,
      scale: Float(pow(Double(dK), -0.5)),
      mask: attnMask
    )

    // Apply dropout after attention
    context = dropout(context)

    // Reshape back
    context = context.transposed(0, 2, 1, 3).reshaped([batchSize, -1, nFeat])

    return linearOut(context)
  }
}
