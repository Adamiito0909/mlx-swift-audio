// Copyright ® Canopy Labs (original model implementation)
// Ported to MLX from https://github.com/canopyai/Orpheus-TTS
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/orpheus.txt

import Foundation
import MLX
import MLXNN

/// NoiseBlock for SNAC decoder - adds learned noise modulation
/// Weight keys: linear.weight_g, linear.weight_v
class SNACNoiseBlock: Module {
  @ModuleInfo var linear: WNConv1d

  init(dim: Int) {
    // WNConv1d for noise modulation - outputs 1 channel for noise scaling
    _linear.wrappedValue = WNConv1d(
      inChannels: dim,
      outChannels: 1,
      kernelSize: 1,
      padding: 0,
      bias: false,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Input shape is likely [N, C, T]
    let B = x.shape[0]
    let T = x.shape[2]

    // Generate noise [B, 1, T]
    let noise = MLXRandom.normal([B, 1, T])

    // Apply the linear transformation
    let h = linear(x)

    // Modulate noise by the linear output and add to input
    let n = noise * h
    return x + n
  }
}
