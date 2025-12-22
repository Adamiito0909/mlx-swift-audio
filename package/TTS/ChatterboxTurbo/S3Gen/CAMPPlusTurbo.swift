// Copyright 2025 Resemble AI (original model implementation)
// Copyright Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// CAM++ speaker encoder for Chatterbox Turbo
// Uses simplified weight key naming (bn instead of nonlinear arrays)

import Foundation
import MLX
import MLXNN

// MARK: - TDNNLayerTurbo

/// Time-Delay Neural Network layer for Turbo
class TDNNLayerTurbo: Module {
  @ModuleInfo var linear: Conv1d
  @ModuleInfo(key: "bn") var bn: BatchNorm

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    bias: Bool = false
  ) {
    var pad = padding
    if padding < 0 {
      precondition(kernelSize % 2 == 1, "Expected odd kernel size")
      pad = (kernelSize - 1) / 2 * dilation
    }

    _linear.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: pad,
      dilation: dilation,
      bias: bias
    )
    _bn.wrappedValue = BatchNorm(featureCount: outChannels)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, C, T) - PyTorch format
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = linear(out)
    out = bn(out)
    out = relu(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }
}

// MARK: - CAMLayerTurbo

/// Context Attentive Module layer for Turbo
class CAMLayerTurbo: Module {
  @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
  @ModuleInfo(key: "linear1") var linear1: Conv1d
  @ModuleInfo(key: "linear2") var linear2: Conv1d

  init(
    bnChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int,
    padding: Int,
    dilation: Int,
    bias: Bool,
    reduction: Int = 2
  ) {
    _linearLocal.wrappedValue = Conv1d(
      inputChannels: bnChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: padding,
      dilation: dilation,
      bias: bias
    )
    _linear1.wrappedValue = Conv1d(
      inputChannels: bnChannels,
      outputChannels: bnChannels / reduction,
      kernelSize: 1
    )
    _linear2.wrappedValue = Conv1d(
      inputChannels: bnChannels / reduction,
      outputChannels: outChannels,
      kernelSize: 1
    )
  }

  /// Segment pooling
  private func segPooling(_ x: MLXArray, segLen: Int = 100) -> MLXArray {
    let shape = x.shape
    let B = shape[0]
    let T = shape[1]
    let C = shape[2]

    let nSegs = (T + segLen - 1) / segLen
    let padLen = nSegs * segLen - T

    var xPadded = x
    if padLen > 0 {
      xPadded = MLX.concatenated([x, MLXArray.zeros([B, padLen, C])], axis: 1)
    }

    let xReshaped = xPadded.reshaped([B, nSegs, segLen, C])
    var seg = MLX.mean(xReshaped, axis: 2)

    seg = seg.expandedDimensions(axis: 2)
    seg = MLX.broadcast(seg, to: [B, nSegs, segLen, C])
    seg = seg.reshaped([B, -1, C])
    seg = seg[0..., 0 ..< T, 0...]

    return seg
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) from caller - convert to (B, T, C) for MLX Conv1d
    let xT = x.swappedAxes(1, 2) // (B, T, C)

    let y = linearLocal(xT)

    // Context: global mean + segment pooling
    var context = MLX.mean(xT, axis: 1, keepDims: true) + segPooling(xT)
    context = relu(linear1(context))
    let m = sigmoid(linear2(context))

    let result = y * m
    // Convert back to (B, C', T')
    return result.swappedAxes(1, 2)
  }
}

// MARK: - CAMDenseTDNNLayerTurbo

/// CAM Dense TDNN layer for Turbo
class CAMDenseTDNNLayerTurbo: Module {
  @ModuleInfo(key: "bn1") var bn1: BatchNorm
  @ModuleInfo(key: "linear1") var linear1: Conv1d
  @ModuleInfo(key: "bn2") var bn2: BatchNorm
  @ModuleInfo(key: "cam_layer") var camLayer: CAMLayerTurbo

  init(
    inChannels: Int,
    outChannels: Int,
    bnChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    bias: Bool = false
  ) {
    precondition(kernelSize % 2 == 1, "Expected odd kernel size")
    let padding = (kernelSize - 1) / 2 * dilation

    _bn1.wrappedValue = BatchNorm(featureCount: inChannels)
    _linear1.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: bnChannels, kernelSize: 1, bias: false)
    _bn2.wrappedValue = BatchNorm(featureCount: bnChannels)
    _camLayer.wrappedValue = CAMLayerTurbo(
      bnChannels: bnChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: padding,
      dilation: dilation,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, C, T) - PyTorch format
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = bn1(out)
    out = relu(out)
    out = linear1(out)
    out = bn2(out)
    out = relu(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    out = camLayer(out)
    return out
  }
}

// MARK: - CAMDenseTDNNBlockTurbo

/// CAM Dense TDNN block with multiple layers for Turbo
class CAMDenseTDNNBlockTurbo: Module {
  @ModuleInfo var layers: [CAMDenseTDNNLayerTurbo]

  init(
    numLayers: Int,
    inChannels: Int,
    outChannels: Int,
    bnChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    bias: Bool = false
  ) {
    var layersList: [CAMDenseTDNNLayerTurbo] = []
    for i in 0 ..< numLayers {
      layersList.append(CAMDenseTDNNLayerTurbo(
        inChannels: inChannels + i * outChannels,
        outChannels: outChannels,
        bnChannels: bnChannels,
        kernelSize: kernelSize,
        stride: stride,
        dilation: dilation,
        bias: bias
      ))
    }
    _layers.wrappedValue = layersList
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x
    for layer in layers {
      out = MLX.concatenated([out, layer(out)], axis: 1)
    }
    return out
  }
}

// MARK: - TransitLayerTurbo

/// Transition layer between dense blocks for Turbo
class TransitLayerTurbo: Module {
  @ModuleInfo(key: "bn") var bn: BatchNorm
  @ModuleInfo var linear: Conv1d

  init(inChannels: Int, outChannels: Int, bias: Bool = true) {
    _bn.wrappedValue = BatchNorm(featureCount: inChannels)
    _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = bn(out)
    out = relu(out)
    out = linear(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    return out
  }
}

// MARK: - DenseLayerTurbo

/// Dense layer for final embedding for Turbo
class DenseLayerTurbo: Module {
  @ModuleInfo var linear: Conv1d
  @ModuleInfo(key: "bn") var bn: BatchNorm

  init(inChannels: Int, outChannels: Int, bias: Bool = false) {
    _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: bias)
    _bn.wrappedValue = BatchNorm(featureCount: outChannels, affine: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out: MLXArray

    if x.ndim == 2 {
      // (B, C) -> (B, 1, C) for Conv1d
      out = x.expandedDimensions(axis: 1)
      out = linear(out)
      out = out.squeezed(axis: 1) // (B, C')
      out = bn(out)
    } else {
      // (B, C, T) -> (B, T, C) for Conv1d
      out = x.swappedAxes(1, 2)
      out = linear(out)
      out = out.swappedAxes(1, 2)
      // BatchNorm expects (B, ..., C)
      out = out.swappedAxes(1, 2)
      out = bn(out)
      out = out.swappedAxes(1, 2)
    }

    return out
  }
}

// MARK: - FCMTurbo

/// Feature Channel Module for Turbo - processes input features
class FCMTurbo: Module {
  var inPlanes: Int
  let outChannels: Int

  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "bn1") var bn1: BatchNorm
  @ModuleInfo(key: "layer1") var layer1: [BasicResBlockTurbo]
  @ModuleInfo(key: "layer2") var layer2: [BasicResBlockTurbo]
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "bn2") var bn2: BatchNorm

  init(mChannels: Int = 32, featDim: Int = 80) {
    inPlanes = mChannels

    _conv1.wrappedValue = Conv2d(
      inputChannels: 1,
      outputChannels: mChannels,
      kernelSize: IntOrPair(3),
      stride: IntOrPair(1),
      padding: IntOrPair(1),
      bias: false
    )
    _bn1.wrappedValue = BatchNorm(featureCount: mChannels)

    // Create layer1
    var l1: [BasicResBlockTurbo] = []
    l1.append(BasicResBlockTurbo(inPlanes: inPlanes, planes: mChannels, stride: 2))
    inPlanes = mChannels
    l1.append(BasicResBlockTurbo(inPlanes: inPlanes, planes: mChannels, stride: 1))
    _layer1.wrappedValue = l1

    // Create layer2
    var l2: [BasicResBlockTurbo] = []
    l2.append(BasicResBlockTurbo(inPlanes: inPlanes, planes: mChannels, stride: 2))
    inPlanes = mChannels
    l2.append(BasicResBlockTurbo(inPlanes: inPlanes, planes: mChannels, stride: 1))
    _layer2.wrappedValue = l2

    _conv2.wrappedValue = Conv2d(
      inputChannels: mChannels,
      outputChannels: mChannels,
      kernelSize: IntOrPair(3),
      stride: IntOrPair((2, 1)),
      padding: IntOrPair(1),
      bias: false
    )
    _bn2.wrappedValue = BatchNorm(featureCount: mChannels)

    outChannels = mChannels * (featDim / 8)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input x: (B, T, F) - add channel dimension
    var out = x.swappedAxes(1, 2) // (B, F, T)
    out = out.expandedDimensions(axis: -1) // (B, F, T, 1) = (B, H, W, C)
    out = relu(bn1(conv1(out)))

    for layer in layer1 {
      out = layer(out)
    }
    for layer in layer2 {
      out = layer(out)
    }

    out = relu(bn2(conv2(out)))

    // Reshape: (B, H, W, C) -> (B, C*H, W)
    let shape = out.shape
    let B = shape[0]
    let H = shape[1]
    let W = shape[2]
    let C = shape[3]

    // Permute to (B, C, H, W) then reshape
    out = out.transposed(0, 3, 1, 2) // (B, C, H, W)
    out = out.reshaped([B, C * H, W])
    return out
  }
}

// MARK: - BasicResBlockTurbo

/// Basic residual block for 2D convolution for Turbo
class BasicResBlockTurbo: Module {
  static let expansion = 1

  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "bn1") var bn1: BatchNorm
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "bn2") var bn2: BatchNorm

  // Shortcut connection (separate keys to match Python MLX)
  let useShortcut: Bool
  @ModuleInfo(key: "shortcut_conv") var shortcutConv: Conv2d?
  @ModuleInfo(key: "shortcut_bn") var shortcutBn: BatchNorm?

  init(inPlanes: Int, planes: Int, stride: Int = 1) {
    useShortcut = stride != 1 || inPlanes != Self.expansion * planes

    _conv1.wrappedValue = Conv2d(
      inputChannels: inPlanes,
      outputChannels: planes,
      kernelSize: IntOrPair(3),
      stride: IntOrPair((stride, 1)),
      padding: IntOrPair(1),
      bias: false
    )
    _bn1.wrappedValue = BatchNorm(featureCount: planes)
    _conv2.wrappedValue = Conv2d(
      inputChannels: planes,
      outputChannels: planes,
      kernelSize: IntOrPair(3),
      stride: IntOrPair(1),
      padding: IntOrPair(1),
      bias: false
    )
    _bn2.wrappedValue = BatchNorm(featureCount: planes)

    if useShortcut {
      _shortcutConv.wrappedValue = Conv2d(
        inputChannels: inPlanes,
        outputChannels: Self.expansion * planes,
        kernelSize: IntOrPair(1),
        stride: IntOrPair((stride, 1)),
        bias: false
      )
      _shortcutBn.wrappedValue = BatchNorm(featureCount: Self.expansion * planes)
    } else {
      _shortcutConv.wrappedValue = nil
      _shortcutBn.wrappedValue = nil
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = relu(bn1(conv1(x)))
    out = bn2(conv2(out))

    let shortcutOut: MLXArray = if useShortcut, let conv = shortcutConv, let bn = shortcutBn {
      bn(conv(x))
    } else {
      x
    }

    out = out + shortcutOut
    return relu(out)
  }
}

// MARK: - StatsPoolTurbo

/// Statistics pooling layer for Turbo
class StatsPoolTurbo: Module {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T)
    let mean = MLX.mean(x, axis: 2, keepDims: false)
    let variance = MLX.variance(x, axis: 2, keepDims: false)
    let std = MLX.sqrt(variance + 1e-5)
    return MLX.concatenated([mean, std], axis: -1)
  }
}

// MARK: - CAMPPlusTurbo

/// CAM++ speaker embedding model for Chatterbox Turbo
class CAMPPlusTurbo: Module {
  @ModuleInfo(key: "head") var head: FCMTurbo
  @ModuleInfo(key: "tdnn") var tdnn: TDNNLayerTurbo
  @ModuleInfo var blocks: [CAMDenseTDNNBlockTurbo]
  @ModuleInfo(key: "transits") var transits: [TransitLayerTurbo]
  @ModuleInfo(key: "out_bn") var outBn: BatchNorm
  @ModuleInfo(key: "stats") var stats: StatsPoolTurbo
  @ModuleInfo(key: "dense") var dense: DenseLayerTurbo

  init(
    featDim: Int = 80,
    embeddingSize: Int = 192,
    growthRate: Int = 32,
    bnSize: Int = 4,
    initChannels: Int = 128
  ) {
    _head.wrappedValue = FCMTurbo(mChannels: 32, featDim: featDim)
    var channels = 32 * (featDim / 8) // head.outChannels

    _tdnn.wrappedValue = TDNNLayerTurbo(
      inChannels: channels,
      outChannels: initChannels,
      kernelSize: 5,
      stride: 2,
      padding: -1,
      dilation: 1
    )
    channels = initChannels

    // Dense TDNN blocks
    var blocksArray: [CAMDenseTDNNBlockTurbo] = []
    var transitsArray: [TransitLayerTurbo] = []

    let configs: [(Int, Int, Int)] = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
    for (numLayers, kernelSize, dilation) in configs {
      blocksArray.append(CAMDenseTDNNBlockTurbo(
        numLayers: numLayers,
        inChannels: channels,
        outChannels: growthRate,
        bnChannels: bnSize * growthRate,
        kernelSize: kernelSize,
        dilation: dilation
      ))
      channels = channels + numLayers * growthRate

      transitsArray.append(TransitLayerTurbo(
        inChannels: channels,
        outChannels: channels / 2,
        bias: false
      ))
      channels /= 2
    }

    _blocks.wrappedValue = blocksArray
    _transits.wrappedValue = transitsArray
    _outBn.wrappedValue = BatchNorm(featureCount: channels)
    _stats.wrappedValue = StatsPoolTurbo()
    _dense.wrappedValue = DenseLayerTurbo(inChannels: channels * 2, outChannels: embeddingSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = head(x)
    out = tdnn(out)

    // Dense blocks with transitions
    for (block, transit) in zip(blocks, transits) {
      out = block(out)
      out = transit(out)
    }

    // Output processing
    out = out.swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    out = outBn(out)
    out = relu(out)
    out = out.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)

    out = stats(out)
    out = dense(out)

    return out
  }

  /// Inference on raw audio waveform
  func inference(_ audio: MLXArray) -> MLXArray {
    var audioData = audio
    if audioData.ndim == 1 {
      audioData = audioData.expandedDimensions(axis: 0)
    }

    // Extract fbank features for each audio in batch
    let features = (0 ..< audioData.shape[0]).map { i -> MLXArray in
      var fbank = kaldiFbankCAMPPlus(audio: audioData[i])
      // Mean normalization
      fbank = fbank - MLX.mean(fbank, axis: 0, keepDims: true)
      return fbank
    }

    // Pad to same length
    let maxLen = features.map { $0.shape[0] }.max() ?? 0
    let paddedFeatures = features.map { f -> MLXArray in
      if f.shape[0] < maxLen {
        let pad = MLXArray.zeros([maxLen - f.shape[0], f.shape[1]])
        return MLX.concatenated([f, pad], axis: 0)
      }
      return f
    }

    // Stack to batch: (B, T, F)
    let batchFeatures = MLX.stacked(paddedFeatures, axis: 0)

    return callAsFunction(batchFeatures)
  }
}
