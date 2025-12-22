// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Conditional decoder for Chatterbox Turbo S3Gen flow matching

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Conv1d PyTorch Format Wrapper

/// Conv1d wrapper that accepts PyTorch format (B, C, T) input
class CBTConv1dPT: Module {
  @ModuleInfo var conv: Conv1d

  init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1) {
    _conv.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: padding,
      dilation: dilation
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) -> (B, T, C)
    var h = x.transposed(0, 2, 1)
    h = conv(h)
    // (B, T, C) -> (B, C, T)
    return h.transposed(0, 2, 1)
  }
}

// MARK: - ConvTranspose1d PyTorch Format Wrapper

/// ConvTranspose1d wrapper that accepts PyTorch format (B, C, T) input
class CBTConvTranspose1dPT: Module {
  var weight: MLXArray
  var bias: MLXArray?

  let padding: Int
  let stride: Int
  let kernelSize: Int

  init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0) {
    let scale: Float = 1.0 / Float(inChannels * kernelSize)
    weight = MLXRandom.uniform(
      low: -scale, high: scale,
      [outChannels, kernelSize, inChannels]
    )
    bias = MLXArray.zeros([outChannels])
    self.padding = padding
    self.stride = stride
    self.kernelSize = kernelSize
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) -> (B, T, C)
    let xNLC = x.transposed(0, 2, 1)
    var y = convTransposed1d(xNLC, weight, stride: stride, padding: padding)
    if let b = bias { y = y + b }
    // (B, T, C) -> (B, C, T)
    return y.transposed(0, 2, 1)
  }
}

// MARK: - Sinusoidal Position Embedding

/// Sinusoidal positional embeddings for timesteps
func sinusoidalPosEmb(timesteps: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
  var t = timesteps
  if t.ndim == 0 {
    t = t.expandedDimensions(axis: 0)
  }

  let halfDim = dim / 2
  let emb = log(10000.0) / Float(halfDim - 1)
  var embeddings = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * -emb)
  embeddings = scale * t.expandedDimensions(axis: 1) * embeddings.expandedDimensions(axis: 0)
  return MLX.concatenated([MLX.sin(embeddings), MLX.cos(embeddings)], axis: -1)
}

// MARK: - Timestep Embedding

/// MLP for timestep embedding
class CBTTimestepEmbedding: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  init(inChannels: Int, timeEmbedDim: Int) {
    _linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
    _linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = linear1(x)
    h = silu(h)
    return linear2(h)
  }
}

// MARK: - Causal Conv1d

/// Causal 1D convolution with left padding
class CBTCausalConv1d: Module {
  let kernelSize: Int
  let dilation: Int
  let causalPadding: Int

  @ModuleInfo var conv: CBTConv1dPT

  init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1) {
    self.kernelSize = kernelSize
    self.dilation = dilation
    causalPadding = (kernelSize - 1) * dilation

    _conv.wrappedValue = CBTConv1dPT(
      inChannels: inChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      dilation: dilation
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T)
    // Pad on left side only for causal
    let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((causalPadding, 0))])
    return conv(padded)
  }
}

// MARK: - Block1D

/// Basic 1D block with conv, group norm, and mish activation
class CBTBlock1D: Module {
  @ModuleInfo var conv: CBTConv1dPT
  @ModuleInfo var norm: GroupNorm

  init(dim: Int, dimOut: Int, groups: Int = 8) {
    _conv.wrappedValue = CBTConv1dPT(inChannels: dim, outChannels: dimOut, kernelSize: 3, padding: 1)
    _norm.wrappedValue = GroupNorm(groupCount: groups, dimensions: dimOut)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
    var h = x * mask
    h = conv(h)
    // GroupNorm expects (..., C), we have (B, C, T)
    h = h.transposed(0, 2, 1) // (B, T, C)
    h = norm(h)
    h = h.transposed(0, 2, 1) // (B, C, T)
    h = mish(h)
    return h * mask
  }
}

// MARK: - CausalBlock1D

/// Causal version of Block1D with LayerNorm
class CBTCausalBlock1D: Module {
  @ModuleInfo(key: "block") var block: [Module]

  init(dim: Int, dimOut: Int, groups _: Int = 8) {
    _block.wrappedValue = [
      CBTCausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3),
      LayerNorm(dimensions: dimOut),
    ]
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
    var h = x * mask
    h = (block[0] as! CBTCausalConv1d)(h)
    // LayerNorm expects (B, T, C), we have (B, C, T)
    h = h.transposed(0, 2, 1)
    h = (block[1] as! LayerNorm)(h)
    h = h.transposed(0, 2, 1)
    h = mish(h)
    return h * mask
  }
}

// MARK: - ResnetBlock1D

/// Resnet block with time embedding
class CBTResnetBlock1D: Module {
  let causal: Bool

  @ModuleInfo var mlp: [Linear]
  @ModuleInfo(key: "block1") var block1: Module
  @ModuleInfo(key: "block2") var block2: Module
  @ModuleInfo(key: "res_conv") var resConv: CBTConv1dPT

  init(dim: Int, dimOut: Int, timeEmbDim: Int, causal: Bool = true, groups: Int = 8) {
    self.causal = causal

    _mlp.wrappedValue = [Linear(timeEmbDim, dimOut)]

    if causal {
      _block1.wrappedValue = CBTCausalBlock1D(dim: dim, dimOut: dimOut, groups: groups)
      _block2.wrappedValue = CBTCausalBlock1D(dim: dimOut, dimOut: dimOut, groups: groups)
    } else {
      _block1.wrappedValue = CBTBlock1D(dim: dim, dimOut: dimOut, groups: groups)
      _block2.wrappedValue = CBTBlock1D(dim: dimOut, dimOut: dimOut, groups: groups)
    }

    _resConv.wrappedValue = CBTConv1dPT(inChannels: dim, outChannels: dimOut, kernelSize: 1)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
    var h: MLXArray = if causal {
      (block1 as! CBTCausalBlock1D)(x, mask: mask)
    } else {
      (block1 as! CBTBlock1D)(x, mask: mask)
    }

    // Apply Mish then linear
    h = h + mlp[0](mish(timeEmb)).expandedDimensions(axis: 2)

    if causal {
      h = (block2 as! CBTCausalBlock1D)(h, mask: mask)
    } else {
      h = (block2 as! CBTBlock1D)(h, mask: mask)
    }

    return h + resConv(x * mask)
  }
}

// MARK: - Downsample1D

/// 1D downsampling layer
class CBTDownsample1D: Module {
  @ModuleInfo var conv: CBTConv1dPT

  init(dim: Int) {
    _conv.wrappedValue = CBTConv1dPT(inChannels: dim, outChannels: dim, kernelSize: 3, stride: 2, padding: 1)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - Upsample1D

/// 1D upsampling layer with conv transpose
class CBTUpsample1DDecoder: Module {
  @ModuleInfo var conv: CBTConvTranspose1dPT

  init(dim: Int) {
    _conv.wrappedValue = CBTConvTranspose1dPT(inChannels: dim, outChannels: dim, kernelSize: 4, stride: 2, padding: 1)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - SelfAttention1D

/// Self-attention for 1D sequences (bidirectional)
class CBTSelfAttention1D: Module {
  let numHeads: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]

  init(dim: Int, numHeads: Int = 8, headDim: Int = 64, dropout _: Float = 0.0) {
    self.numHeads = numHeads
    self.headDim = headDim
    let innerDim = numHeads * headDim
    scale = pow(Float(headDim), -0.5)

    _toQ.wrappedValue = Linear(dim, innerDim, bias: false)
    _toK.wrappedValue = Linear(dim, innerDim, bias: false)
    _toV.wrappedValue = Linear(dim, innerDim, bias: false)
    _toOut.wrappedValue = [Linear(innerDim, dim)]
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
    let B = x.shape[0]
    let T = x.shape[1]

    var q = toQ(x)
    var k = toK(x)
    var v = toV(x)

    // Reshape for multi-head attention
    q = q.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
    k = k.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
    v = v.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)

    // Convert padding mask to additive attention mask for SDPA
    // mask: (B, T) with 1 for valid, 0 for padding -> additive: 0 for valid, -inf for padding
    var attnMask: MLXArray? = nil
    if let mask {
      let maskExpanded = mask.expandedDimensions(axes: [1, 2]) // (B, 1, 1, T)
      attnMask = MLX.where(maskExpanded .> 0, MLXArray(Float(0)), MLXArray(-Float.infinity))
    }

    // Use optimized scaled dot product attention (bidirectional)
    var out = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: attnMask
    )

    // Reshape back
    out = out.transposed(0, 2, 1, 3).reshaped([B, T, -1])
    return toOut[0](out)
  }
}

// MARK: - GELU with Projection

/// GELU activation with linear projection
class CBTGELUProj: Module {
  @ModuleInfo var proj: Linear

  init(dimIn: Int, dimOut: Int) {
    _proj.wrappedValue = Linear(dimIn, dimOut)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    gelu(proj(x))
  }
}

// MARK: - FeedForward Net Container

/// Container for FeedForward net components (GELU and output Linear)
/// Weight keys are remapped in sanitizeWeights: net.0.* -> net.gelu.*, net.1.* -> net.linear.*
class CBTFFNet: Module {
  @ModuleInfo(key: "gelu") var gelu: CBTGELUProj
  @ModuleInfo var linear: Linear

  init(dim: Int, innerDim: Int) {
    _gelu.wrappedValue = CBTGELUProj(dimIn: dim, dimOut: innerDim)
    _linear.wrappedValue = Linear(innerDim, dim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    linear(gelu(x))
  }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation
class CBTDecoderFeedForward: Module {
  @ModuleInfo(key: "net") var net: CBTFFNet

  init(dim: Int, mult: Int = 4, dropout _: Float = 0.0) {
    let innerDim = dim * mult
    _net.wrappedValue = CBTFFNet(dim: dim, innerDim: innerDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    net(x)
  }
}

// MARK: - TransformerBlock

/// Basic transformer block with self-attention and feed-forward
class CBTTransformerBlock: Module {
  @ModuleInfo(key: "attn1") var attn1: CBTSelfAttention1D
  @ModuleInfo(key: "ff") var ff: CBTDecoderFeedForward
  @ModuleInfo(key: "norm1") var norm1: LayerNorm
  @ModuleInfo(key: "norm3") var norm3: LayerNorm

  init(dim: Int, numHeads: Int = 8, headDim: Int = 64, ffMult: Int = 4, dropout: Float = 0.0) {
    _attn1.wrappedValue = CBTSelfAttention1D(dim: dim, numHeads: numHeads, headDim: headDim, dropout: dropout)
    _ff.wrappedValue = CBTDecoderFeedForward(dim: dim, mult: ffMult, dropout: dropout)
    _norm1.wrappedValue = LayerNorm(dimensions: dim)
    _norm3.wrappedValue = LayerNorm(dimensions: dim)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
    var h = x + attn1(norm1(x), mask: mask)
    h = h + ff(norm3(h))
    return h
  }
}

// MARK: - DownBlock

/// Down block containing ResNet, transformers, and downsample
class CBTDownBlock: Module {
  let isLast: Bool
  let causal: Bool

  @ModuleInfo(key: "resnet") var resnet: CBTResnetBlock1D
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [CBTTransformerBlock]
  @ModuleInfo(key: "downsample") var downsample: Module

  init(
    inputChannel: Int,
    outputChannel: Int,
    timeEmbedDim: Int,
    causal: Bool,
    nBlocks: Int,
    numHeads: Int,
    attentionHeadDim: Int,
    isLast: Bool
  ) {
    self.isLast = isLast
    self.causal = causal

    _resnet.wrappedValue = CBTResnetBlock1D(dim: inputChannel, dimOut: outputChannel, timeEmbDim: timeEmbedDim, causal: causal)
    _transformerBlocks.wrappedValue = (0 ..< nBlocks).map { _ in
      CBTTransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
    }

    if isLast {
      _downsample.wrappedValue = causal
        ? CBTCausalConv1d(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
        : CBTConv1dPT(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3, padding: 1)
    } else {
      _downsample.wrappedValue = CBTDownsample1D(dim: outputChannel)
    }
  }
}

// MARK: - MidBlock

/// Mid block containing ResNet and transformers
class CBTMidBlock: Module {
  @ModuleInfo(key: "resnet") var resnet: CBTResnetBlock1D
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [CBTTransformerBlock]

  init(
    channels: Int,
    timeEmbedDim: Int,
    causal: Bool,
    nBlocks: Int,
    numHeads: Int,
    attentionHeadDim: Int
  ) {
    _resnet.wrappedValue = CBTResnetBlock1D(dim: channels, dimOut: channels, timeEmbDim: timeEmbedDim, causal: causal)
    _transformerBlocks.wrappedValue = (0 ..< nBlocks).map { _ in
      CBTTransformerBlock(dim: channels, numHeads: numHeads, headDim: attentionHeadDim)
    }
  }
}

// MARK: - UpBlock

/// Up block containing ResNet, transformers, and upsample
class CBTUpBlock: Module {
  let isLast: Bool
  let causal: Bool

  @ModuleInfo(key: "resnet") var resnet: CBTResnetBlock1D
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [CBTTransformerBlock]
  @ModuleInfo(key: "upsample") var upsample: Module

  init(
    inputChannel: Int,
    outputChannel: Int,
    timeEmbedDim: Int,
    causal: Bool,
    nBlocks: Int,
    numHeads: Int,
    attentionHeadDim: Int,
    isLast: Bool
  ) {
    self.isLast = isLast
    self.causal = causal

    _resnet.wrappedValue = CBTResnetBlock1D(dim: inputChannel, dimOut: outputChannel, timeEmbDim: timeEmbedDim, causal: causal)
    _transformerBlocks.wrappedValue = (0 ..< nBlocks).map { _ in
      CBTTransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
    }

    if isLast {
      _upsample.wrappedValue = causal
        ? CBTCausalConv1d(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
        : CBTConv1dPT(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3, padding: 1)
    } else {
      _upsample.wrappedValue = CBTUpsample1DDecoder(dim: outputChannel)
    }
  }
}

// MARK: - ConditionalDecoder

/// Conditional decoder for flow matching (U-Net style architecture)
class CBTConditionalDecoder: Module {
  let inChannels: Int
  let outChannels: Int
  let causal: Bool
  let meanflow: Bool

  @ModuleInfo(key: "time_mlp") var timeMlp: CBTTimestepEmbedding
  @ModuleInfo(key: "down_blocks") var downBlocks: [CBTDownBlock]
  @ModuleInfo(key: "mid_blocks") var midBlocks: [CBTMidBlock]
  @ModuleInfo(key: "up_blocks") var upBlocks: [CBTUpBlock]
  @ModuleInfo(key: "final_block") var finalBlock: Module
  @ModuleInfo(key: "final_proj") var finalProj: CBTConv1dPT
  @ModuleInfo(key: "time_embed_mixer") var timeEmbedMixer: Linear?

  init(
    inChannels: Int = 320,
    outChannels: Int = 80,
    causal: Bool = true,
    channels: [Int] = [256],
    dropout _: Float = 0.0,
    attentionHeadDim: Int = 64,
    nBlocks: Int = 4,
    numMidBlocks: Int = 12,
    numHeads: Int = 8,
    meanflow: Bool = false
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.causal = causal
    self.meanflow = meanflow

    // Time embedding
    let timeEmbedDim = channels[0] * 4
    _timeMlp.wrappedValue = CBTTimestepEmbedding(inChannels: inChannels, timeEmbedDim: timeEmbedDim)

    // Down blocks
    var outputChannel = inChannels
    var downBlocksArray: [CBTDownBlock] = []
    for (i, ch) in channels.enumerated() {
      let inputChannel = outputChannel
      outputChannel = ch
      let isLast = i == channels.count - 1

      downBlocksArray.append(CBTDownBlock(
        inputChannel: inputChannel,
        outputChannel: outputChannel,
        timeEmbedDim: timeEmbedDim,
        causal: causal,
        nBlocks: nBlocks,
        numHeads: numHeads,
        attentionHeadDim: attentionHeadDim,
        isLast: isLast
      ))
    }
    _downBlocks.wrappedValue = downBlocksArray

    // Mid blocks
    _midBlocks.wrappedValue = (0 ..< numMidBlocks).map { _ in
      CBTMidBlock(
        channels: channels.last!,
        timeEmbedDim: timeEmbedDim,
        causal: causal,
        nBlocks: nBlocks,
        numHeads: numHeads,
        attentionHeadDim: attentionHeadDim
      )
    }

    // Up blocks
    let channelsUp = channels.reversed() + [channels[0]]
    var upBlocksArray: [CBTUpBlock] = []
    for i in 0 ..< (channelsUp.count - 1) {
      let inputChannel = channelsUp[i] * 2 // Skip connection
      let outChannel = channelsUp[i + 1]
      let isLast = i == channelsUp.count - 2

      upBlocksArray.append(CBTUpBlock(
        inputChannel: inputChannel,
        outputChannel: outChannel,
        timeEmbedDim: timeEmbedDim,
        causal: causal,
        nBlocks: nBlocks,
        numHeads: numHeads,
        attentionHeadDim: attentionHeadDim,
        isLast: isLast
      ))
    }
    _upBlocks.wrappedValue = upBlocksArray

    // Final layers
    let finalCh = channelsUp.last!
    if causal {
      _finalBlock.wrappedValue = CBTCausalBlock1D(dim: finalCh, dimOut: finalCh)
    } else {
      _finalBlock.wrappedValue = CBTBlock1D(dim: finalCh, dimOut: finalCh)
    }
    _finalProj.wrappedValue = CBTConv1dPT(inChannels: finalCh, outChannels: outChannels, kernelSize: 1)

    // Meanflow time mixing
    if meanflow {
      _timeEmbedMixer.wrappedValue = Linear(timeEmbedDim * 2, timeEmbedDim, bias: false)
    }
  }

  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray,
    mu: MLXArray,
    t: MLXArray,
    spks: MLXArray? = nil,
    cond: MLXArray? = nil,
    r: MLXArray? = nil
  ) -> MLXArray {
    // Time embedding
    var tEmb = sinusoidalPosEmb(timesteps: t, dim: inChannels)
    tEmb = timeMlp(tEmb)

    if meanflow, let r, let mixer = timeEmbedMixer {
      var rEmb = sinusoidalPosEmb(timesteps: r, dim: inChannels)
      rEmb = timeMlp(rEmb)
      let concatEmb = MLX.concatenated([tEmb, rEmb], axis: -1)
      tEmb = mixer(concatEmb)
    }

    // Concatenate inputs: x, mu, spks, cond
    var inputs = [x, mu]
    if let spks {
      let spksExpanded = MLX.broadcast(
        spks.expandedDimensions(axis: 2),
        to: [spks.shape[0], spks.shape[1], x.shape[2]]
      )
      inputs.append(spksExpanded)
    }
    if let cond {
      inputs.append(cond)
    }

    var h = MLX.concatenated(inputs, axis: 1)

    // Down path
    var hiddens: [MLXArray] = []
    var masks = [mask]
    for downBlock in downBlocks {
      let maskDown = masks.last!
      h = downBlock.resnet(h, mask: maskDown, timeEmb: tEmb)

      // Transpose for transformer: (B, C, T) -> (B, T, C)
      h = h.transposed(0, 2, 1)
      let maskT = maskDown.squeezed(axis: 1)
      for block in downBlock.transformerBlocks {
        h = block(h, mask: maskT)
      }
      h = h.transposed(0, 2, 1)

      hiddens.append(h)

      // Apply downsample
      if downBlock.isLast {
        if downBlock.causal {
          h = (downBlock.downsample as! CBTCausalConv1d)(h * maskDown)
        } else {
          h = (downBlock.downsample as! CBTConv1dPT)(h * maskDown)
        }
      } else {
        h = (downBlock.downsample as! CBTDownsample1D)(h * maskDown)
      }

      // Downsample mask
      masks.append(maskDown[0..., 0..., .stride(by: 2)])
    }

    masks.removeLast()
    let maskMid = masks.last!

    // Mid path
    for midBlock in midBlocks {
      h = midBlock.resnet(h, mask: maskMid, timeEmb: tEmb)
      h = h.transposed(0, 2, 1)
      let maskT = maskMid.squeezed(axis: 1)
      for block in midBlock.transformerBlocks {
        h = block(h, mask: maskT)
      }
      h = h.transposed(0, 2, 1)
    }

    // Up path
    var currentMask = masks.removeLast()
    for (i, upBlock) in upBlocks.enumerated() {
      let maskUp = i < masks.count ? masks.removeLast() : currentMask
      currentMask = maskUp
      let skip = hiddens.removeLast()

      // Align sizes
      h = h[0..., 0..., 0 ..< skip.shape[2]]
      h = MLX.concatenated([h, skip], axis: 1)

      h = upBlock.resnet(h, mask: maskUp, timeEmb: tEmb)
      h = h.transposed(0, 2, 1)
      let maskT = maskUp.squeezed(axis: 1)
      for block in upBlock.transformerBlocks {
        h = block(h, mask: maskT)
      }
      h = h.transposed(0, 2, 1)

      // Apply upsample
      if upBlock.isLast {
        if upBlock.causal {
          h = (upBlock.upsample as! CBTCausalConv1d)(h * maskUp)
        } else {
          h = (upBlock.upsample as! CBTConv1dPT)(h * maskUp)
        }
      } else {
        h = (upBlock.upsample as! CBTUpsample1DDecoder)(h * maskUp)
      }
    }

    // Final
    if causal {
      h = (finalBlock as! CBTCausalBlock1D)(h, mask: currentMask)
    } else {
      h = (finalBlock as! CBTBlock1D)(h, mask: currentMask)
    }
    h = finalProj(h * currentMask)

    return h * mask
  }
}
