// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLX
import MLXNN

class AlbertEncoder: Module {
  let config: AlbertConfig

  @ModuleInfo(key: "embedding_hidden_mapping_in") var embeddingHiddenMappingIn: Linear
  @ModuleInfo(key: "albert_layer_groups") var albertLayerGroups: [AlbertLayerGroup]

  init(config: AlbertConfig) {
    self.config = config

    _embeddingHiddenMappingIn.wrappedValue = Linear(config.embeddingSize, config.hiddenSize)
    _albertLayerGroups.wrappedValue = (0 ..< config.numHiddenGroups).map { _ in
      AlbertLayerGroup(config: config)
    }
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil,
  ) -> MLXArray {
    var output = embeddingHiddenMappingIn(hiddenStates)

    for i in 0 ..< config.numHiddenLayers {
      let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)

      output = albertLayerGroups[groupIdx](output, attentionMask: attentionMask)
    }

    return output
  }
}
