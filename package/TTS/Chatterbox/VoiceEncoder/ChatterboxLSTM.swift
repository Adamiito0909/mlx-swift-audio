// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// Multi-layer unidirectional LSTM for Chatterbox VoiceEncoder.
// Uses MLXNN.LSTM layers with weight keys: layers.N.Wx, layers.N.Wh, layers.N.bias

import Foundation
import MLX
import MLXNN

// MARK: - ChatterboxLSTM

/// Multi-layer unidirectional LSTM using MLXNN.LSTM
///
/// Weight naming follows MLX convention:
/// - layers.0.Wx, layers.0.Wh, layers.0.bias for layer 0
/// - layers.1.Wx, layers.1.Wh, layers.1.bias for layer 1
/// - layers.2.Wx, layers.2.Wh, layers.2.bias for layer 2
class ChatterboxLSTM: Module {
  let inputSize: Int
  let hiddenSize: Int
  let numLayers: Int

  @ModuleInfo var layers: [LSTM]

  init(inputSize: Int, hiddenSize: Int, numLayers: Int = 3) {
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.numLayers = numLayers

    // Create layers: first layer takes inputSize, rest take hiddenSize
    var layerArray: [LSTM] = []
    for i in 0 ..< numLayers {
      let layerInputSize = i == 0 ? inputSize : hiddenSize
      layerArray.append(LSTM(inputSize: layerInputSize, hiddenSize: hiddenSize))
    }
    _layers.wrappedValue = layerArray
  }

  /// Process sequence through all LSTM layers
  func callAsFunction(
    _ x: MLXArray,
    hidden: (MLXArray, MLXArray)? = nil,
  ) -> (MLXArray, (MLXArray, MLXArray)) {
    var hList: [MLXArray?]
    var cList: [MLXArray?]

    if let (h0, c0) = hidden {
      hList = (0 ..< numLayers).map { i in h0[i] }
      cList = (0 ..< numLayers).map { i in c0[i] }
    } else {
      hList = [MLXArray?](repeating: nil, count: numLayers)
      cList = [MLXArray?](repeating: nil, count: numLayers)
    }

    var currentOutput = x
    var finalHidden: [MLXArray] = []
    var finalCell: [MLXArray] = []

    for i in 0 ..< numLayers {
      let (allH, allC) = layers[i](currentOutput, hidden: hList[i], cell: cList[i])
      currentOutput = allH

      // Extract final timestep for h and c
      let seqLen = allH.dim(-2)
      finalHidden.append(allH[.ellipsis, seqLen - 1, 0...])
      finalCell.append(allC[.ellipsis, seqLen - 1, 0...])
    }

    let hN = MLX.stacked(finalHidden, axis: 0)
    let cN = MLX.stacked(finalCell, axis: 0)

    return (currentOutput, (hN, cN))
  }
}
