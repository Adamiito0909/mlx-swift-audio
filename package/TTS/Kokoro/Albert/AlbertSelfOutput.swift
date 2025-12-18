// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXNN

class AlbertSelfOutput {
  let dense: Linear
  let layerNorm: LayerNorm

  init(config: AlbertConfig) {
    dense = Linear(config.hiddenSize, config.hiddenSize)
    layerNorm = LayerNorm(
      dimensions: config.hiddenSize,
      eps: config.layerNormEps,
    )
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    inputTensor: MLXArray,
  ) -> MLXArray {
    var output = dense(hiddenStates)
    output = layerNorm(output + inputTensor)
    return output
  }
}
