// Copyright © 2025 Resemble AI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/resemble-ai/chatterbox
// License: licenses/chatterbox.txt

// S3Gen for Chatterbox Turbo: Speech tokens to waveform

import Foundation
import MLX
import MLXNN

// MARK: - Constants

/// S3Gen constants
enum S3GenTurboConstants {
  static let s3GenSR: Int = 24000 // Output sample rate
  static let s3SR: Int = 16000 // Input tokenizer sample rate
  static let s3GenSil: Int = 4299 // Silence token
  static let speechVocabSize: Int = 6561
}

// MARK: - Helper Functions

/// Remove tokens outside valid vocabulary
func dropInvalidTokensTurbo(_ x: MLXArray) -> MLXArray {
  x[x .< S3GenTurboConstants.speechVocabSize]
}

// MARK: - S3Token2Mel

/// S3Gen's CFM decoder: maps S3 speech tokens to mel-spectrograms
class S3Token2MelTurbo: Module {
  let meanflow: Bool
  let tokenMelRatio: Int = 2
  let preLookaheadLen: Int = 3

  @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
  @ModuleInfo(key: "speaker_encoder") var speakerEncoder: CAMPPlusTurbo
  @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
  @ModuleInfo var encoder: CBTUpsampleEncoder
  @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
  @ModuleInfo var decoder: CBTCausalConditionalCFM

  init(meanflow: Bool = false) {
    self.meanflow = meanflow

    // Token embedding
    _inputEmbedding.wrappedValue = Embedding(embeddingCount: S3GenTurboConstants.speechVocabSize, dimensions: 512)

    // Speaker encoder (CAMPPlusTurbo)
    _speakerEncoder.wrappedValue = CAMPPlusTurbo(
      featDim: 80,
      embeddingSize: 192,
      growthRate: 32,
      bnSize: 4,
      initChannels: 128
    )

    // Speaker embedding projection
    _spkEmbedAffineLayer.wrappedValue = Linear(192, 80)

    // Encoder
    _encoder.wrappedValue = CBTUpsampleEncoder(
      inputSize: 512,
      outputSize: 512,
      attentionHeads: 8,
      linearUnits: 2048,
      numBlocks: 6,
      dropoutRate: 0.1
    )
    _encoderProj.wrappedValue = Linear(512, 80)

    // CFM decoder/estimator
    let estimator = CBTConditionalDecoder(
      inChannels: 320,
      outChannels: 80,
      causal: true,
      channels: [256],
      dropout: 0.0,
      attentionHeadDim: 64,
      nBlocks: 4,
      numMidBlocks: 12,
      numHeads: 8,
      meanflow: meanflow
    )

    // Flow matching
    _decoder.wrappedValue = CBTCausalConditionalCFM(
      inChannels: 240,
      spkEmbDim: 80,
      sigmaMin: 1e-6,
      tScheduler: "cosine",
      inferenceCfgRate: 0.7,
      estimator: estimator
    )

    super.init()
  }

  /// Generate mel-spectrogram from speech tokens
  func callAsFunction(
    speechTokens: MLXArray,
    refDict: CBTRefDict,
    nCfmTimesteps: Int? = nil,
    finalize: Bool = true
  ) -> MLXArray {
    let B = speechTokens.shape[0]

    // Get reference data
    var promptToken = refDict.promptToken
    let promptTokenLen = refDict.promptTokenLen
    var promptFeat = refDict.promptFeat
    var embedding = refDict.embedding

    // Broadcast reference data if needed
    if promptToken.shape[0] != B {
      promptToken = MLX.broadcast(promptToken, to: [B] + Array(promptToken.shape.dropFirst()))
    }
    if embedding.shape[0] != B {
      embedding = MLX.broadcast(embedding, to: [B] + Array(embedding.shape.dropFirst()))
    }
    if promptFeat.shape[0] != B {
      promptFeat = MLX.broadcast(promptFeat, to: [B] + Array(promptFeat.shape.dropFirst()))
    }

    // Speaker embedding projection with normalization
    let norm = MLX.sqrt(MLX.sum(embedding * embedding, axis: -1, keepDims: true)) + 1e-8
    embedding = embedding / norm
    embedding = spkEmbedAffineLayer(embedding)

    // Concatenate prompt and input tokens
    let tokenLen = MLXArray([Int32(speechTokens.shape[1])] + Array(repeating: Int32(speechTokens.shape[1]), count: B - 1))
    let token = MLX.concatenated([promptToken, speechTokens], axis: 1)
    let totalTokenLen = promptTokenLen + tokenLen

    // Create mask
    let maxLen = token.shape[1]
    let mask = MLXArray(0 ..< maxLen).expandedDimensions(axis: 0) .< totalTokenLen.expandedDimensions(axis: 1)
    let maskFloat = mask.expandedDimensions(axis: 2).asType(.float32)

    // Embed tokens
    let tokenEmb = inputEmbedding(token.asType(.int32)) * maskFloat

    // Encode
    var (h, hMasks) = encoder(tokenEmb, xsLens: totalTokenLen)

    // Handle non-finalized chunks
    if !finalize {
      let truncLen = preLookaheadLen * tokenMelRatio
      h = h[0..., 0 ..< (h.shape[1] - truncLen), 0...]
    }

    let hLengths = MLX.sum(hMasks.squeezed(axis: 1).asType(.int32), axis: -1)
    let melLen1 = promptFeat.shape[1]
    let melLen2 = h.shape[1] - melLen1
    h = encoderProj(h)

    // Prepare conditioning
    let zerosPadding = MLXArray.zeros([B, melLen2, 80])
    var conds = MLX.concatenated([promptFeat, zerosPadding], axis: 1)
    conds = conds.transposed(0, 2, 1) // (B, 80, T)

    // Mask for decoder
    let decoderMask = MLXArray(0 ..< h.shape[1]).expandedDimensions(axis: 0) .< hLengths.expandedDimensions(axis: 1)
    let decoderMaskFloat = decoderMask.expandedDimensions(axis: 1).asType(.float32)

    // Default timesteps
    let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)

    // Generate noise for meanflow
    var noisedMels: MLXArray? = nil
    if meanflow {
      noisedMels = MLXRandom.normal([B, 80, speechTokens.shape[1] * 2])
    }

    // Flow matching
    let feat = decoder(
      mu: h.transposed(0, 2, 1),
      mask: decoderMaskFloat,
      nTimesteps: timesteps,
      spks: embedding,
      cond: conds,
      noisedMels: noisedMels,
      meanflow: meanflow
    )

    // Remove prompt portion
    return feat[0..., 0..., melLen1...]
  }
}

// MARK: - S3Token2Wav

/// Full S3Gen: speech tokens to waveform
/// Combines token-to-mel (CFM) and mel-to-wav (HiFiGAN)
class S3Token2WavTurbo: S3Token2MelTurbo {
  @ModuleInfo(key: "mel2wav") var mel2wav: HiFTGeneratorTurbo
  var trimFade: MLXArray

  override init(meanflow: Bool = false) {
    // HiFiGAN vocoder
    _mel2wav.wrappedValue = HiFTGeneratorTurbo(
      samplingRate: S3GenTurboConstants.s3GenSR,
      upsampleRates: [8, 5, 3],
      upsampleKernelSizes: [16, 11, 7],
      sourceResblockKernelSizes: [7, 7, 11],
      sourceResblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    // Trim fade for artifact reduction
    let nTrim = S3GenTurboConstants.s3GenSR / 50 // 20ms
    var trimFadeValues = [Float](repeating: 0, count: 2 * nTrim)
    for i in nTrim ..< (2 * nTrim) {
      let t = Float(i - nTrim) / Float(nTrim)
      trimFadeValues[i] = (cos(Float.pi * (1 - t)) + 1) / 2
    }
    trimFade = MLXArray(trimFadeValues)

    super.init(meanflow: meanflow)
  }

  /// Full inference: speech tokens to waveform
  func inference(
    speechTokens: MLXArray,
    refDict: CBTRefDict,
    nCfmTimesteps: Int? = nil
  ) -> (MLXArray, MLXArray) {
    // Default timesteps for meanflow
    let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)

    // Generate mel
    let outputMels = callAsFunction(
      speechTokens: speechTokens,
      refDict: refDict,
      nCfmTimesteps: timesteps,
      finalize: true
    )

    // Vocoder - transpose for HiFiGAN
    let melT = outputMels.transposed(0, 2, 1) // (B, T, 80)
    let (outputWavs, outputSources) = mel2wav.inference(melT, cacheSource: nil)

    // Apply trim fade to reduce artifacts
    var result = outputWavs
    let fadeLen = trimFade.shape[0]
    if result.shape[1] >= fadeLen {
      let fadedStart = result[0..., 0 ..< fadeLen] * trimFade
      result = MLX.concatenated([fadedStart, result[0..., fadeLen...]], axis: 1)
    }

    return (result, outputSources)
  }

  /// Streaming inference: convert speech tokens to waveform
  func inferenceStream(
    speechTokens: MLXArray,
    refDict: CBTRefDict,
    nCfmTimesteps: Int? = nil,
    prevAudioSamples: Int = 0,
    isFinal: Bool = false
  ) -> (MLXArray, Int) {
    // Default timesteps for meanflow
    let timesteps = nCfmTimesteps ?? (meanflow ? 2 : 10)

    // Generate mel from all accumulated tokens
    let outputMels = callAsFunction(
      speechTokens: speechTokens,
      refDict: refDict,
      nCfmTimesteps: timesteps,
      finalize: isFinal
    )

    // Vocoder
    let melT = outputMels.transposed(0, 2, 1)
    var (outputWavs, _) = mel2wav.inference(melT, cacheSource: nil)

    // Apply trim fade only on first chunk
    if prevAudioSamples == 0 {
      let fadeLen = trimFade.shape[0]
      if outputWavs.shape[1] >= fadeLen {
        let fadedStart = outputWavs[0..., 0 ..< fadeLen] * trimFade
        outputWavs = MLX.concatenated([fadedStart, outputWavs[0..., fadeLen...]], axis: 1)
      }
    }

    let totalSamples = outputWavs.shape[1]

    // Return only new samples
    let newAudio: MLXArray = if prevAudioSamples > 0, prevAudioSamples < totalSamples {
      outputWavs[0..., prevAudioSamples...]
    } else if prevAudioSamples == 0 {
      outputWavs
    } else {
      outputWavs[0..., 0 ..< 0] // Empty
    }

    return (newAudio, totalSamples)
  }
}

// MARK: - Reference Dictionary

/// Container for reference embedding data
struct CBTRefDict: @unchecked Sendable {
  var promptToken: MLXArray
  var promptTokenLen: MLXArray
  var promptFeat: MLXArray
  var promptFeatLen: MLXArray
  var embedding: MLXArray

  init(
    promptToken: MLXArray,
    promptTokenLen: MLXArray,
    promptFeat: MLXArray,
    promptFeatLen: MLXArray,
    embedding: MLXArray
  ) {
    self.promptToken = promptToken
    self.promptTokenLen = promptTokenLen
    self.promptFeat = promptFeat
    self.promptFeatLen = promptFeatLen
    self.embedding = embedding
  }
}

// MARK: - Alias

/// Alias for compatibility
typealias S3GenTurbo = S3Token2WavTurbo
