// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

//  Feed forward module for Conformer encoder

import Foundation
import MLX
import MLXNN

// MARK: - PositionwiseFeedForward

/// Positionwise feed forward layer
/// Feed forward applied on each position of the sequence.
/// The output dim is same as the input dim.
class PositionwiseFeedForward: Module {
  @ModuleInfo(key: "w_1") var w1: Linear
  @ModuleInfo(key: "w_2") var w2: Linear
  let activation: UnaryLayer
  let dropoutRate: Float

  init(
    idim: Int,
    hiddenUnits: Int,
    dropoutRate: Float,
    activation: UnaryLayer? = nil,
  ) {
    _w1.wrappedValue = Linear(idim, hiddenUnits)
    _w2.wrappedValue = Linear(hiddenUnits, idim)
    self.activation = activation ?? ReLU()
    self.dropoutRate = dropoutRate
  }

  func callAsFunction(_ xs: MLXArray) -> MLXArray {
    var x = w1(xs)
    x = activation(x)
    // Note: Dropout would be applied here during training
    return w2(x)
  }
}
