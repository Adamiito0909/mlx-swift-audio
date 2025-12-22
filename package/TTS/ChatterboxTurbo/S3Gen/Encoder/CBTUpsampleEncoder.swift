// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Upsampling Conformer encoder for Chatterbox Turbo S3Gen

import Foundation
import MLX
import MLXNN

// MARK: - Relative Positional Encoding (ESPnet Style)

/// Relative positional encoding module (ESPnet style)
/// Creates positional encodings for both positive and negative relative positions
class CBTRelPositionalEncoding: Module {
  let dModel: Int
  let xscale: Float
  let maxLen: Int
  var pe: MLXArray?

  init(dModel: Int, dropout _: Float = 0.1, maxLen: Int = 5000) {
    self.dModel = dModel
    xscale = sqrt(Float(dModel))
    self.maxLen = maxLen
    pe = nil
    super.init()
    extendPE(maxLen)
  }

  private func extendPE(_ size: Int) {
    if let pe, pe.shape[1] >= size * 2 - 1 {
      return
    }

    // Create positive and negative position encodings
    let position = MLXArray(0 ..< size).asType(.float32).expandedDimensions(axis: 1)
    let divTerm = MLX.exp(
      MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32) *
        (-log(10000.0) / Float(dModel))
    )

    // Positive positions - interleave sin/cos
    let pePositiveSin = MLX.sin(position * divTerm)
    let pePositiveCos = MLX.cos(position * divTerm)
    let pePositive = MLX.concatenated(
      [pePositiveSin.expandedDimensions(axis: -1), pePositiveCos.expandedDimensions(axis: -1)],
      axis: -1
    ).reshaped([size, dModel])

    // Negative positions
    let peNegativeSin = MLX.sin(-position * divTerm)
    let peNegativeCos = MLX.cos(-position * divTerm)
    let peNegative = MLX.concatenated(
      [peNegativeSin.expandedDimensions(axis: -1), peNegativeCos.expandedDimensions(axis: -1)],
      axis: -1
    ).reshaped([size, dModel])

    // Flip positive and concatenate: [pos_reversed, neg[1:]]
    let indices = MLXArray((0 ..< size).reversed())
    let pePositiveFlipped = pePositive[indices]
    let peNegativeTail = peNegative[1...]

    let fullPe = MLX.concatenated([pePositiveFlipped, peNegativeTail], axis: 0)
    pe = fullPe.expandedDimensions(axis: 0) // (1, 2*size-1, d_model)
  }

  func callAsFunction(_ x: MLXArray, offset _: Int = 0) -> (MLXArray, MLXArray) {
    let T = x.shape[1]
    extendPE(T)

    // Scale input by sqrt(d_model)
    let scaledX = x * xscale

    // Get positional embedding centered around current sequence
    guard let pe else {
      fatalError("PE not initialized")
    }
    let center = pe.shape[1] / 2
    let posEmb = pe[0..., (center - T + 1) ..< (center + T), 0...]

    return (scaledX, posEmb)
  }
}

// MARK: - Linear Input

/// Linear input projection with LayerNorm and positional encoding
class CBTLinearInput: Module {
  @ModuleInfo var linear: Linear
  @ModuleInfo var norm: LayerNorm
  @ModuleInfo(key: "pos_enc") var posEnc: CBTRelPositionalEncoding

  init(inputSize: Int, outputSize: Int, dropout: Float = 0.1) {
    _linear.wrappedValue = Linear(inputSize, outputSize)
    _norm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)
    _posEnc.wrappedValue = CBTRelPositionalEncoding(dModel: outputSize, dropout: dropout)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    var h = linear(x)
    h = norm(h)
    let (scaledH, posEmb) = posEnc(h)
    return (scaledH, posEmb, mask)
  }
}

// MARK: - Relative Position Multi-Head Attention

/// Multi-head self-attention with relative positional encoding
class CBTRelPositionAttention: Module {
  let nHead: Int
  let dK: Int
  let scale: Float

  @ModuleInfo(key: "linear_q") var linearQ: Linear
  @ModuleInfo(key: "linear_k") var linearK: Linear
  @ModuleInfo(key: "linear_v") var linearV: Linear
  @ModuleInfo(key: "linear_out") var linearOut: Linear
  @ModuleInfo(key: "linear_pos") var linearPos: Linear

  // Learnable positional bias parameters
  @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
  @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

  init(nHead: Int, nFeat: Int, dropoutRate _: Float = 0.0, keyBias: Bool = true) {
    self.nHead = nHead
    dK = nFeat / nHead
    scale = pow(Float(dK), -0.5)

    _linearQ.wrappedValue = Linear(nFeat, nFeat)
    _linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
    _linearV.wrappedValue = Linear(nFeat, nFeat)
    _linearOut.wrappedValue = Linear(nFeat, nFeat)
    _linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)

    _posBiasU.wrappedValue = MLXArray.zeros([nHead, dK])
    _posBiasV.wrappedValue = MLXArray.zeros([nHead, dK])

    super.init()
  }

  private func relShift(_ x: MLXArray) -> MLXArray {
    let B = x.shape[0]
    let nHeads = x.shape[1]
    let T1 = x.shape[2]
    let T2 = x.shape[3]

    // Pad with zeros on the left
    let zeroPad = MLXArray.zeros([B, nHeads, T1, 1])
    var xPadded = MLX.concatenated([zeroPad, x], axis: -1)

    // Reshape and extract the valid part
    xPadded = xPadded.reshaped([B, nHeads, T2 + 1, T1])
    var result = xPadded[0..., 0..., 1..., 0...] // Remove first row
    result = result.reshaped([B, nHeads, T1, T2])

    // Keep only positions 0 to time1
    return result[0..., 0..., 0..., 0 ..< (T2 / 2 + 1)]
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray?,
    posEmb: MLXArray?
  ) -> MLXArray {
    let B = x.shape[0]
    let T = x.shape[1]
    let D = x.shape[2]

    // Compute Q, K, V
    let q = linearQ(x).reshaped([B, T, nHead, dK])
    let k = linearK(x).reshaped([B, T, nHead, dK]).transposed(0, 2, 1, 3)
    let v = linearV(x).reshaped([B, T, nHead, dK]).transposed(0, 2, 1, 3)

    // Content attention: (q + pos_bias_u) @ k^T
    let qWithBiasU = (q + posBiasU).transposed(0, 2, 1, 3) // (B, n_head, T, d_k)
    let matrixAC = qWithBiasU.matmul(k.transposed(0, 1, 3, 2)) // (B, n_head, T, T)

    // Position attention
    var scores: MLXArray
    if let posEmb {
      let TPos = posEmb.shape[1]
      var p = linearPos(posEmb)
      p = p.reshaped([1, TPos, nHead, dK]).transposed(0, 2, 1, 3)

      let qWithBiasV = (q + posBiasV).transposed(0, 2, 1, 3)
      var matrixBD = qWithBiasV.matmul(p.transposed(0, 1, 3, 2))

      // Apply relative shift when shapes don't match
      if matrixAC.shape != matrixBD.shape {
        matrixBD = relShift(matrixBD)
      }

      scores = (matrixAC + matrixBD) * scale
    } else {
      scores = matrixAC * scale
    }

    // Apply mask if provided
    if let mask {
      let maskExpanded: MLXArray = if mask.ndim == 2 {
        mask[0..., .newAxis, .newAxis, 0...]
      } else {
        mask[0..., .newAxis, 0..., 0...]
      }
      scores = MLX.where(maskExpanded .> 0, scores, MLXArray(-Float.infinity))
    }

    var attn = softmax(scores, axis: -1)
    // Replace NaN from softmax(-inf) with 0
    attn = MLX.where(MLX.isNaN(attn), MLXArray(0.0), attn)

    var out = attn.matmul(v)
    out = out.transposed(0, 2, 1, 3).reshaped([B, T, D])

    return linearOut(out)
  }
}

// MARK: - Positionwise Feed Forward

/// Position-wise feed-forward network with SiLU activation
class CBTFeedForward: Module {
  @ModuleInfo(key: "w_1") var w1: Linear
  @ModuleInfo(key: "w_2") var w2: Linear

  init(dModel: Int, dInner: Int, dropout _: Float = 0.1) {
    _w1.wrappedValue = Linear(dModel, dInner)
    _w2.wrappedValue = Linear(dInner, dModel)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    w2(silu(w1(x)))
  }
}

// MARK: - Conformer Encoder Layer

/// Single Conformer encoder layer with pre-norm style
class CBTEncoderLayer: Module {
  let size: Int

  @ModuleInfo(key: "norm_mha") var normMha: LayerNorm
  @ModuleInfo(key: "self_attn") var selfAttn: CBTRelPositionAttention
  @ModuleInfo(key: "norm_ff") var normFf: LayerNorm
  @ModuleInfo(key: "feed_forward") var feedForward: CBTFeedForward

  init(size: Int, nHead: Int, dInner: Int, dropoutRate: Float = 0.1, keyBias: Bool = true) {
    self.size = size
    _normMha.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
    _selfAttn.wrappedValue = CBTRelPositionAttention(
      nHead: nHead, nFeat: size, dropoutRate: dropoutRate, keyBias: keyBias
    )
    _normFf.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
    _feedForward.wrappedValue = CBTFeedForward(dModel: size, dInner: dInner, dropout: dropoutRate)
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray?,
    posEmb: MLXArray?
  ) -> MLXArray {
    // Multi-head self-attention with pre-norm and residual
    var residual = x
    var h = normMha(x)
    h = residual + selfAttn(h, mask: mask, posEmb: posEmb)

    // Feed-forward with pre-norm and residual
    residual = h
    h = normFf(h)
    h = residual + feedForward(h)

    return h
  }
}

// MARK: - Pre-Lookahead Layer

/// Pre-lookahead convolution layer
class CBTPreLookaheadLayer: Module {
  let channels: Int
  let preLookaheadLen: Int

  @ModuleInfo(key: "conv1") var conv1: Conv1d
  @ModuleInfo(key: "conv2") var conv2: Conv1d

  init(channels: Int, preLookaheadLen: Int = 3) {
    self.channels = channels
    self.preLookaheadLen = preLookaheadLen

    _conv1.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: preLookaheadLen + 1,
      stride: 1,
      padding: 0
    )
    _conv2.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 1,
      padding: 0
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // MLX Conv1d expects (B, T, C) format
    // Look ahead padding on time dimension (axis 1)
    var out = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, preLookaheadLen)), IntOrPair(0)])
    out = leakyRelu(conv1(out))

    // Causal padding for second conv
    out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((2, 0)), IntOrPair(0)])
    out = conv2(out)

    // Residual connection
    return out + x
  }
}

// MARK: - Upsample 1D Encoder

/// 1D upsampling layer for encoder
class CBTUpsample1D: Module {
  let channels: Int
  let stride: Int

  @ModuleInfo var conv: Conv1d

  init(channels: Int, stride: Int = 2) {
    self.channels = channels
    self.stride = stride

    _conv.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: stride * 2 + 1,
      stride: 1,
      padding: 0
    )
  }

  func callAsFunction(_ x: MLXArray, xLens: MLXArray) -> (MLXArray, MLXArray) {
    // Nearest neighbor upsampling on time dimension
    var upsampled = MLX.repeated(x, count: stride, axis: 1)

    // Causal padding on time dimension and conv
    upsampled = MLX.padded(upsampled, widths: [IntOrPair(0), IntOrPair((stride * 2, 0)), IntOrPair(0)])
    upsampled = conv(upsampled)

    return (upsampled, xLens * stride)
  }
}

// MARK: - Upsample Conformer Encoder

/// Upsampling Conformer encoder for S3Gen
/// Converts speech tokens to mel-spectrogram features
class CBTUpsampleEncoder: Module {
  let outputSizeValue: Int

  @ModuleInfo(key: "embed") var embed: CBTLinearInput
  @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: CBTPreLookaheadLayer
  @ModuleInfo(key: "encoders") var encoders: [CBTEncoderLayer]
  @ModuleInfo(key: "up_layer") var upLayer: CBTUpsample1D
  @ModuleInfo(key: "up_embed") var upEmbed: CBTLinearInput
  @ModuleInfo(key: "up_encoders") var upEncoders: [CBTEncoderLayer]
  @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

  init(
    inputSize: Int = 512,
    outputSize: Int = 512,
    attentionHeads: Int = 8,
    linearUnits: Int = 2048,
    numBlocks: Int = 6,
    dropoutRate: Float = 0.1
  ) {
    outputSizeValue = outputSize

    // Input embedding
    _embed.wrappedValue = CBTLinearInput(inputSize: inputSize, outputSize: outputSize, dropout: dropoutRate)

    // Pre-lookahead layer
    _preLookaheadLayer.wrappedValue = CBTPreLookaheadLayer(channels: outputSize, preLookaheadLen: 3)

    // Encoder layers
    _encoders.wrappedValue = (0 ..< numBlocks).map { _ in
      CBTEncoderLayer(size: outputSize, nHead: attentionHeads, dInner: linearUnits, dropoutRate: dropoutRate)
    }

    // Upsampling
    _upLayer.wrappedValue = CBTUpsample1D(channels: outputSize, stride: 2)

    // Post-upsample embedding
    _upEmbed.wrappedValue = CBTLinearInput(inputSize: inputSize, outputSize: outputSize, dropout: dropoutRate)

    // Post-upsample encoder layers
    _upEncoders.wrappedValue = (0 ..< 4).map { _ in
      CBTEncoderLayer(size: outputSize, nHead: attentionHeads, dInner: linearUnits, dropoutRate: dropoutRate)
    }

    // Final norm
    _afterNorm.wrappedValue = LayerNorm(dimensions: outputSize)
  }

  var outputSize: Int { outputSizeValue }

  func callAsFunction(_ xs: MLXArray, xsLens: MLXArray) -> (MLXArray, MLXArray) {
    let T = xs.shape[1]

    // Create mask
    var mask = MLXArray(0 ..< T).expandedDimensions(axis: 0) .< xsLens.expandedDimensions(axis: 1)
    mask = mask.expandedDimensions(axis: 1) // (B, 1, T)

    // Input projection
    var (h, posEmb, _) = embed(xs, mask: mask)

    // Pre-lookahead
    h = preLookaheadLayer(h)

    // Encoder layers
    let mask1D = mask.squeezed(axis: 1) // (B, T)
    for layer in encoders {
      h = layer(h, mask: mask1D, posEmb: posEmb)
    }

    // Upsampling
    var newLens: MLXArray
    (h, newLens) = upLayer(h, xLens: xsLens)

    // Update mask
    let T2 = h.shape[1]
    mask = MLXArray(0 ..< T2).expandedDimensions(axis: 0) .< newLens.expandedDimensions(axis: 1)
    mask = mask.expandedDimensions(axis: 1)

    // Post-upsample embedding
    (h, posEmb, _) = upEmbed(h, mask: mask)

    // Post-upsample encoder
    let mask1D2 = mask.squeezed(axis: 1)
    for layer in upEncoders {
      h = layer(h, mask: mask1D2, posEmb: posEmb)
    }

    // Final norm
    h = afterNorm(h)

    return (h, mask)
  }
}
