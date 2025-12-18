// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX
import MLXNN

/// Full SenseVoice encoder with three encoder stacks
///
/// Architecture:
/// - Scale input by sqrt(output_size)
/// - encoders0: 1 layer, processes input from 560 -> 512 dims
/// - encoders: 49 layers, main encoder at 512 dims
/// - after_norm: LayerNorm before time-pooling
/// - tp_encoders: 20 layers, time-pooling encoder at 512 dims
/// - tp_norm: Final LayerNorm
///
/// The encoder uses SANM (Self-Attention with Memory) blocks which
/// combine standard attention with FSMN for local context modeling.
class SenseVoiceEncoder: Module {
  let config: SenseVoiceEncoderConfig
  let outputSize: Int

  // Initial encoder layer(s) - handles input dimension change
  @ModuleInfo var encoders0: [EncoderLayerSANM]

  // Main encoder layers
  @ModuleInfo var encoders: [EncoderLayerSANM]

  // Time-pooling encoder layers
  @ModuleInfo(key: "tp_encoders") var tpEncoders: [EncoderLayerSANM]

  // Normalization layers
  @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm
  @ModuleInfo(key: "tp_norm") var tpNorm: LayerNorm

  /// Initialize the SenseVoice encoder
  ///
  /// - Parameter config: Encoder configuration
  init(config: SenseVoiceEncoderConfig) {
    self.config = config
    outputSize = config.encoderDim

    // Initial encoder layer(s) - handles input dimension change (560 -> 512)
    _encoders0.wrappedValue = (0 ..< config.numEncoders0).map { i in
      EncoderLayerSANM(
        inSize: i == 0 ? config.inputDim : config.encoderDim,
        size: config.encoderDim,
        nHead: config.numHeads,
        dFF: config.ffnDim,
        kernelSize: config.kernelSize,
        sanmShift: config.sanmShift,
        dropoutRate: config.dropout
      )
    }

    // Main encoder layers (49 layers at 512 dims)
    _encoders.wrappedValue = (0 ..< config.numEncoders).map { _ in
      EncoderLayerSANM(
        inSize: config.encoderDim,
        size: config.encoderDim,
        nHead: config.numHeads,
        dFF: config.ffnDim,
        kernelSize: config.kernelSize,
        sanmShift: config.sanmShift,
        dropoutRate: config.dropout
      )
    }

    // Time-pooling encoder layers (20 layers at 512 dims)
    _tpEncoders.wrappedValue = (0 ..< config.numTPEncoders).map { _ in
      EncoderLayerSANM(
        inSize: config.encoderDim,
        size: config.encoderDim,
        nHead: config.numHeads,
        dFF: config.ffnDim,
        kernelSize: config.kernelSize,
        sanmShift: config.sanmShift,
        dropoutRate: config.dropout
      )
    }

    // Normalization layers
    _afterNorm.wrappedValue = LayerNorm(dimensions: config.encoderDim)
    _tpNorm.wrappedValue = LayerNorm(dimensions: config.encoderDim)
  }

  /// Forward pass through the encoder
  ///
  /// - Parameters:
  ///   - x: LFR-processed audio features (batch, seq, inputDim)
  ///   - lengths: Optional sequence lengths for each batch item
  /// - Returns: Tuple of (encoder output, output lengths)
  ///   - Encoder output: (batch, seq, encoderDim)
  ///   - Output lengths: Same as input lengths
  func callAsFunction(_ x: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let (batchSize, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

    let actualLengths: MLXArray = if let lengths {
      lengths
    } else {
      MLXArray(Array(repeating: Int32(seqLen), count: batchSize))
    }

    // Scale input by sqrt(output_size) - matches original
    var out = x * Float(sqrt(Double(outputSize)))

    // No mask needed for full attention
    let mask: MLXArray? = nil

    // Initial encoder(s)
    for layer in encoders0 {
      out = layer(out, mask: mask)
    }

    // Main encoder
    for layer in encoders {
      out = layer(out, mask: mask)
    }

    // Apply after_norm
    out = afterNorm(out)

    // Time-pooling encoder
    for layer in tpEncoders {
      out = layer(out, mask: mask)
    }

    // Final normalization
    out = tpNorm(out)

    return (out, actualLengths)
  }
}
