// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXNN

class UpSample1d {
  private let layerType: String
  private let interpolate: Upsample

  init(layerType: String) {
    self.layerType = layerType
    interpolate = Upsample(
      scaleFactor: 2.0,
      mode: .nearest,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if layerType == "none" {
      x
    } else {
      interpolate(x)
    }
  }
}
