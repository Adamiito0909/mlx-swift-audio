// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXFast
import MLXNN

class LayerNormInference: Module {
  let eps: Float
  @ParameterInfo(key: "gamma") var weight: MLXArray?
  @ParameterInfo(key: "beta") var bias: MLXArray?

  init(dims: Int = 0, eps: Float = 1e-5) {
    self.eps = eps
    _weight.wrappedValue = dims > 0 ? MLXArray.ones([dims]) : nil
    _bias.wrappedValue = dims > 0 ? MLXArray.zeros([dims]) : nil
  }

  open func callAsFunction(_ x: MLXArray) -> MLXArray {
    MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
  }
}
