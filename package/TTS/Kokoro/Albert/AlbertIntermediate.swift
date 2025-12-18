// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXNN

class AlbertIntermediate {
  let dense: Linear

  init(config: AlbertConfig) {
    dense = Linear(config.hiddenSize, config.intermediateSize)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var output = dense(hiddenStates)
    output = MLXNN.gelu(output)
    return output
  }
}
