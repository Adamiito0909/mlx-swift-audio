// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

/// Main Fun-ASR model combining audio encoder, adaptor, and LLM decoder
class FunASRModel: Module {
  let config: FunASRConfig

  @ModuleInfo(key: "audio_encoder") var audioEncoder: SenseVoiceEncoder
  @ModuleInfo(key: "audio_adaptor") var audioAdaptor: AudioAdaptor
  var llm: Qwen3ForCausalLM

  // Path to the model directory (for loading tokenizer)
  var modelDirectory: URL?

  init(config: FunASRConfig) {
    self.config = config

    // Audio encoder
    _audioEncoder.wrappedValue = SenseVoiceEncoder(config: config.encoder)

    // Audio adaptor
    _audioAdaptor.wrappedValue = AudioAdaptor(config: config.adaptor)

    // LLM decoder
    llm = Qwen3ForCausalLM(config: config.llm)
  }

  /// Encode audio to embeddings
  ///
  /// - Parameter audio: Audio waveform (T,) at 16kHz
  /// - Returns: Audio embeddings projected to LLM space (1, seq, llmDim)
  func encodeAudio(_ audio: MLXArray) -> MLXArray {
    // Preprocess audio
    let features = preprocessAudio(audio, nMels: config.nMels, lfrM: config.lfrM, lfrN: config.lfrN)

    // Add batch dimension
    var batchedFeatures = features
    if features.ndim == 2 {
      batchedFeatures = features.expandedDimensions(axis: 0)
    }

    // Encode
    let (encoderOut, lengths) = audioEncoder(batchedFeatures, lengths: nil)

    // Adapt to LLM space
    let (adapted, _) = audioAdaptor(encoderOut, lengths: lengths)

    return adapted
  }

  /// Merge audio embeddings with text embeddings
  ///
  /// The audio embeddings replace the speech token placeholder region
  /// (between SOS and EOS tokens in the prompt).
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs with speech placeholders (batch, seq)
  ///   - audioEmbeddings: Encoded audio embeddings (batch, audioSeq, llmDim)
  ///   - sosTokenId: Start of speech token ID
  ///   - eosTokenId: End of speech token ID
  /// - Returns: Combined embeddings (batch, newSeq, llmDim)
  func mergeEmbeddings(
    inputIds: MLXArray,
    audioEmbeddings: MLXArray,
    sosTokenId: Int,
    eosTokenId: Int
  ) -> MLXArray {
    // Get text embeddings
    let textEmbeddings = llm.getInputEmbeddings()(inputIds)

    let batchSize = inputIds.shape[0]
    var allEmbeddings: [MLXArray] = []

    for b in 0 ..< batchSize {
      let ids = inputIds[b]
      let textEmb = textEmbeddings[b]
      let audioEmb = audioEmbeddings.ndim == 3 ? audioEmbeddings[b] : audioEmbeddings

      // Find SOS and EOS positions
      // Use a simple loop since MLX doesn't have efficient argwhere
      var sosPos: Int?
      var eosPos: Int?

      let idsArray = ids.asArray(Int32.self)
      for (i, id) in idsArray.enumerated() {
        if id == Int32(sosTokenId), sosPos == nil {
          sosPos = i
        }
        if id == Int32(eosTokenId), eosPos == nil {
          eosPos = i
        }
      }

      guard let sos = sosPos, let eos = eosPos else {
        // If tokens not found, just use text embeddings
        allEmbeddings.append(textEmb)
        continue
      }

      // Build merged embeddings:
      // [text_before_sos, sos_emb, audio_emb, eos_emb, text_after_eos]
      var parts: [MLXArray] = []

      // Text before SOS (including SOS)
      if sos >= 0 {
        parts.append(textEmb[0 ... sos])
      }

      // Audio embeddings
      parts.append(audioEmb)

      // Text from EOS onwards (including EOS)
      if eos < textEmb.shape[0] {
        parts.append(textEmb[eos...])
      }

      let merged = MLX.concatenated(parts, axis: 0)
      allEmbeddings.append(merged)
    }

    // Pad to same length
    let maxLen = allEmbeddings.map { $0.shape[0] }.max() ?? 0
    let padded: [MLXArray] = allEmbeddings.map { emb in
      if emb.shape[0] < maxLen {
        let padding = MLXArray.zeros([maxLen - emb.shape[0], emb.shape[1]])
        return MLX.concatenated([emb, padding], axis: 0)
      }
      return emb
    }

    return MLX.stacked(padded, axis: 0)
  }

  /// Sample next token from logits
  ///
  /// - Parameters:
  ///   - logits: Logits from model (batch, seq, vocab)
  ///   - temperature: Sampling temperature (0 for greedy)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - topK: Top-k sampling (0 to disable)
  /// - Returns: Sampled token IDs (batch,)
  func sampleNextToken(
    _ logits: MLXArray,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0
  ) -> MLXArray {
    // Get logits for last position - flatten to 1D for single batch
    var lastLogits = logits[0, -1, 0...]

    if temperature == 0 {
      // Greedy decoding
      return MLX.argMax(lastLogits, axis: -1)
    }

    // Apply temperature
    lastLogits = lastLogits / temperature

    // Apply top-k using argPartition
    let effectiveTopK = topK > 0 ? topK : 50

    // Get top-k indices
    let topKIndices = MLX.argPartition(-lastLogits, kth: effectiveTopK - 1)[0 ..< effectiveTopK]
    let topKLogits = lastLogits[topKIndices]

    // Apply top-p within top-k if specified
    var finalLogits = topKLogits
    var finalIndices = topKIndices

    if topP < 1.0 {
      let probs = MLX.softmax(topKLogits)
      let sortedIndices = MLX.argSort(-probs)
      let sortedProbs = probs[sortedIndices]
      let cumsumProbs = MLX.cumsum(sortedProbs)

      // Find cutoff
      let belowThreshold = cumsumProbs .< topP
      let nTokens = max(1, Int(MLX.sum(belowThreshold).item(Int32.self)) + 1)

      let selectedSortedIndices = sortedIndices[0 ..< nTokens]
      finalIndices = topKIndices[selectedSortedIndices]
      finalLogits = lastLogits[finalIndices]
    }

    // Sample from filtered distribution
    let probs = MLX.softmax(finalLogits)
    let idx = MLXRandom.categorical(MLX.log(probs + 1e-10))

    return finalIndices[idx]
  }

  /// Sanitize weights for loading
  ///
  /// Handles Conv1d weight transposition and key remapping.
  ///
  /// - Parameter weights: Raw weights dictionary
  /// - Returns: Sanitized weights
  static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (k, var v) in weights {
      // Handle FSMN conv weights (PyTorch: [out, 1, kernel] -> MLX: [out, kernel, 1])
      if k.contains("fsmn_block"), k.contains("weight") {
        // Check if transposition is needed
        if v.ndim == 3, v.shape[1] == 1 {
          // Squeeze middle dimension and add at end: (out, 1, kernel) -> (out, kernel, 1)
          v = v.squeezed(axis: 1).expandedDimensions(axis: -1)
        }
      }
      // Handle other conv weights that might need transposition
      else if k.contains("conv"), k.contains("weight") {
        if v.ndim == 3 {
          // Check shape to determine if transposition needed
          if v.shape[2] < v.shape[1] {
            v = v.swappedAxes(-1, -2)
          }
        }
      }

      sanitized[k] = v
    }

    return sanitized
  }

  /// Load Fun-ASR model from HuggingFace Hub
  ///
  /// - Parameters:
  ///   - variant: Model variant to load
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized model with loaded weights
  static func load(
    variant: FunASRModelVariant = .nano4bit,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> FunASRModel {
    let repoId = variant.repoId
    Log.model.info("Loading Fun-ASR from \(repoId)...")

    // Download model files from HuggingFace
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: repoId,
      matching: [
        "*.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
      ],
      progressHandler: progressHandler
    )

    // Load config
    let configURL = modelDirectory.appending(path: "config.json")
    var config: FunASRConfig = if FileManager.default.fileExists(atPath: configURL.path) {
      try FunASRConfig.load(from: configURL)
    } else {
      // Use defaults
      FunASRConfig()
    }

    // Initialize model
    let model = FunASRModel(config: config)

    // Find and load weights
    let weightFiles = try FileManager.default.contentsOfDirectory(
      at: modelDirectory,
      includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "safetensors" }

    guard !weightFiles.isEmpty else {
      throw STTError.modelUnavailable("No safetensors files found in \(repoId)")
    }

    var allWeights: [String: MLXArray] = [:]
    for weightFile in weightFiles {
      let weights = try MLX.loadArrays(url: weightFile)
      allWeights.merge(weights) { _, new in new }
    }

    // Sanitize weights (handle conv transposition)
    let sanitizedWeights = sanitize(allWeights)

    // Check if model is quantized
    let isQuantized = sanitizedWeights.keys.contains { $0.contains(".scales") }
    if isQuantized {
      Log.model.info("Detected quantized Fun-ASR model weights")
      quantize(model: model) { path, _ in
        if sanitizedWeights["\(path).scales"] != nil {
          return (64, 4, .affine)
        }
        return nil
      }
    }

    // Load weights into model
    let parameters = ModuleParameters.unflattened(sanitizedWeights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set to eval mode
    model.train(false)

    // Evaluate to ensure weights are loaded
    eval(model)

    // Store model directory for tokenizer
    model.modelDirectory = modelDirectory

    Log.model.info("Fun-ASR model loaded successfully")

    return model
  }
}
