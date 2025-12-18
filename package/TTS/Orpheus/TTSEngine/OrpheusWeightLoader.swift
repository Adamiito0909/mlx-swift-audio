// Copyright ® Canopy Labs (original model implementation)
// Ported to MLX from https://github.com/canopyai/Orpheus-TTS
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/orpheus.txt

import Foundation
import Hub
import MLX
import MLXNN

class OrpheusWeightLoader {
  private init() {}

  static let defaultRepoId = "mlx-community/orpheus-3b-0.1-ft-4bit"
  static let defaultWeightsFilename = "model.safetensors"

  static func loadWeights(
    repoId: String = defaultRepoId,
    filename: String = defaultWeightsFilename,
    progressHandler: @escaping (Progress) -> Void = { _ in },
  ) async throws -> [String: MLXArray] {
    let modelDirectoryURL = try await HubConfiguration.shared.snapshot(
      from: repoId,
      matching: [filename],
      progressHandler: progressHandler
    )
    let weightFileURL = modelDirectoryURL.appending(path: filename)
    return try loadWeights(from: weightFileURL)
  }

  static func loadWeights(from url: URL) throws -> [String: MLXArray] {
    // Load weights directly without dequantization
    // Quantized models have .weight (uint32 packed), .scales, and .biases
    // These will be loaded into QuantizedLinear layers by the Module system
    try MLX.loadArrays(url: url)
  }
}
