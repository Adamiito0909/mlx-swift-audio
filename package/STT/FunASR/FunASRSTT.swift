// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX

/// Actor wrapper for Fun-ASR model that provides thread-safe transcription
actor FunASRSTT {
  // MARK: - Properties

  // Model and tokenizer are nonisolated(unsafe) because they contain non-Sendable types
  // (MLXArray) but are only accessed within the actor's methods
  nonisolated(unsafe) let model: FunASRModel
  nonisolated(unsafe) let tokenizer: FunASRTokenizer

  // MARK: - Initialization

  private init(model: FunASRModel, tokenizer: FunASRTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  /// Load FunASRSTT from HuggingFace Hub
  ///
  /// - Parameters:
  ///   - variant: Model variant to load
  ///   - progressHandler: Optional callback for download/load progress
  /// - Returns: Initialized FunASRSTT instance
  static func load(
    variant: FunASRModelVariant = .nano4bit,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> FunASRSTT {
    // Load model
    let model = try await FunASRModel.load(
      variant: variant,
      progressHandler: progressHandler
    )

    // Load tokenizer from model directory
    guard let modelDirectory = model.modelDirectory else {
      throw STTError.modelUnavailable("Model directory not set after loading")
    }

    let tokenizer = try await FunASRTokenizer.load(
      from: modelDirectory,
      config: model.config
    )

    return FunASRSTT(model: model, tokenizer: tokenizer)
  }

  // MARK: - Transcription

  /// Transcribe audio to text
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - language: Source language (or .auto for detection)
  ///   - task: Task type (transcribe or translate)
  ///   - targetLanguage: Target language for translation
  ///   - temperature: Sampling temperature (0.0 for greedy)
  ///   - topP: Top-p (nucleus) sampling threshold
  ///   - topK: Top-k sampling (0 to disable)
  ///   - maxTokens: Maximum tokens to generate
  ///   - initialPrompt: Custom instructions or context
  /// - Returns: Transcription result
  func transcribe(
    audio: MLXArray,
    language: FunASRLanguage = .auto,
    task: FunASRTask = .transcribe,
    targetLanguage: FunASRLanguage = .english,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
    maxTokens: Int? = nil,
    initialPrompt: String? = nil
  ) -> TranscriptionResult {
    let startTime = CFAbsoluteTimeGetCurrent()
    let actualMaxTokens = maxTokens ?? model.config.maxTokens

    // Encode audio
    let audioEmbeddings = model.encodeAudio(audio)
    eval(audioEmbeddings)

    // Build prompt and get token IDs
    let promptTokenIds = tokenizer.buildPrompt(
      task: task,
      language: language,
      targetLanguage: targetLanguage,
      initialPrompt: initialPrompt
    )

    // Convert to MLXArray
    let inputIds = MLXArray(promptTokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

    // Merge audio embeddings with prompt
    guard let sosId = tokenizer.sosTokenId, let eosId = tokenizer.eosTokenId else {
      Log.model.error("Special token IDs not found in tokenizer")
      return TranscriptionResult(
        text: "",
        language: language.rawValue,
        segments: [],
        processingTime: CFAbsoluteTimeGetCurrent() - startTime,
        duration: Double(audio.shape[0]) / Double(FunASRAudio.sampleRate)
      )
    }

    var inputEmbeddings = model.mergeEmbeddings(
      inputIds: inputIds,
      audioEmbeddings: audioEmbeddings,
      sosTokenId: sosId,
      eosTokenId: eosId
    )

    // Generate tokens
    var tokens: [Int] = []
    var cache: [(MLXArray, MLXArray)?]? = nil

    for _ in 0 ..< actualMaxTokens {
      let (logits, newCache) = model.llm(
        inputIds: nil,
        inputEmbeddings: inputEmbeddings,
        mask: nil,
        cache: cache
      )
      cache = newCache

      // Sample next token
      let token = model.sampleNextToken(logits, temperature: temperature, topP: topP, topK: topK)
      let tokenId = token.item(Int.self)

      // Check for EOS
      if tokenizer.isEosToken(tokenId) {
        break
      }

      tokens.append(tokenId)

      // Prepare next input (just the new token embedding)
      inputEmbeddings = model.llm.getInputEmbeddings()(token.expandedDimensions(axis: 0).expandedDimensions(axis: 0))
    }

    // Decode tokens
    var text = tokenizer.decode(tokens)
    text = tokenizer.cleanOutput(text)

    // Calculate timing
    let endTime = CFAbsoluteTimeGetCurrent()
    let duration = Double(audio.shape[0]) / Double(FunASRAudio.sampleRate)

    // Clear GPU memory
    MLX.Memory.clearCache()

    return TranscriptionResult(
      text: text,
      language: detectLanguageFromText(text, default: language),
      segments: [], // LLM-based model doesn't produce word-level timestamps
      processingTime: endTime - startTime,
      duration: duration
    )
  }

  /// Stream tokens during generation
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - language: Source language
  ///   - task: Task type
  ///   - targetLanguage: Target language for translation
  ///   - temperature: Sampling temperature
  ///   - topP: Top-p sampling threshold
  ///   - topK: Top-k sampling
  ///   - maxTokens: Maximum tokens to generate
  ///   - initialPrompt: Custom instructions
  /// - Returns: AsyncThrowingStream yielding token IDs
  func generateStreaming(
    audio: MLXArray,
    language: FunASRLanguage = .auto,
    task: FunASRTask = .transcribe,
    targetLanguage: FunASRLanguage = .english,
    temperature: Float = 0.0,
    topP: Float = 0.95,
    topK: Int = 0,
    maxTokens: Int? = nil,
    initialPrompt: String? = nil
  ) -> AsyncThrowingStream<Int, Error> {
    AsyncThrowingStream { continuation in
      Task { [self] in
        do {
          let actualMaxTokens = maxTokens ?? model.config.maxTokens

          // Encode audio
          let audioEmbeddings = model.encodeAudio(audio)
          eval(audioEmbeddings)

          // Build prompt
          let promptTokenIds = tokenizer.buildPrompt(
            task: task,
            language: language,
            targetLanguage: targetLanguage,
            initialPrompt: initialPrompt
          )

          let inputIds = MLXArray(promptTokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

          guard let sosId = tokenizer.sosTokenId, let eosId = tokenizer.eosTokenId else {
            throw STTError.invalidArgument("Special token IDs not found")
          }

          var inputEmbeddings = model.mergeEmbeddings(
            inputIds: inputIds,
            audioEmbeddings: audioEmbeddings,
            sosTokenId: sosId,
            eosTokenId: eosId
          )

          var cache: [(MLXArray, MLXArray)?]? = nil

          for _ in 0 ..< actualMaxTokens {
            try Task.checkCancellation()

            let (logits, newCache) = model.llm(
              inputIds: nil,
              inputEmbeddings: inputEmbeddings,
              mask: nil,
              cache: cache
            )
            cache = newCache

            let token = model.sampleNextToken(logits, temperature: temperature, topP: topP, topK: topK)
            let tokenId = token.item(Int.self)

            if tokenizer.isEosToken(tokenId) {
              break
            }

            continuation.yield(tokenId)

            inputEmbeddings = model.llm.getInputEmbeddings()(token.expandedDimensions(axis: 0).expandedDimensions(axis: 0))
          }

          MLX.Memory.clearCache()
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }

  /// Decode a single token to text
  ///
  /// - Parameter tokenId: Token ID to decode
  /// - Returns: Decoded text
  func decodeToken(_ tokenId: Int) -> String {
    tokenizer.decode([tokenId])
  }

  // MARK: - Private Helpers

  /// Detect language from output text using character heuristics
  private func detectLanguageFromText(_ text: String, default defaultLang: FunASRLanguage) -> String {
    if defaultLang != .auto {
      return defaultLang.rawValue
    }

    guard !text.isEmpty else {
      return "unknown"
    }

    // Character-based heuristics
    let cjkCount = text.unicodeScalars.filter { $0.value >= 0x4E00 && $0.value <= 0x9FFF }.count
    let japaneseCount = text.unicodeScalars.filter { $0.value >= 0x3040 && $0.value <= 0x30FF }.count
    let koreanCount = text.unicodeScalars.filter { $0.value >= 0xAC00 && $0.value <= 0xD7AF }.count
    let arabicCount = text.unicodeScalars.filter { $0.value >= 0x0600 && $0.value <= 0x06FF }.count
    let thaiCount = text.unicodeScalars.filter { $0.value >= 0x0E00 && $0.value <= 0x0E7F }.count
    let cyrillicCount = text.unicodeScalars.filter { $0.value >= 0x0400 && $0.value <= 0x04FF }.count

    let total = text.count
    guard total > 0 else { return "unknown" }

    let totalFloat = Float(total)

    if Float(japaneseCount) / totalFloat > 0.1 { return "ja" }
    if Float(koreanCount) / totalFloat > 0.1 { return "ko" }
    if Float(cjkCount) / totalFloat > 0.2 { return "zh" }
    if Float(arabicCount) / totalFloat > 0.2 { return "ar" }
    if Float(thaiCount) / totalFloat > 0.2 { return "th" }
    if Float(cyrillicCount) / totalFloat > 0.2 { return "ru" }

    return "en"
  }
}
