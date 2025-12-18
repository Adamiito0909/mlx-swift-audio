// Copyright © Anthony DePasquale

import Foundation
import MLX
import MLXAudio

/// Manages STT engine lifecycle and state for multiple providers
@MainActor
@Observable
final class EngineManager {
  // MARK: - Engines

  /// Whisper engine (lazy-loaded when selected)
  private(set) var whisperEngine: WhisperEngine?

  /// Fun-ASR engine (lazy-loaded when selected)
  private(set) var funASREngine: FunASREngine?

  // MARK: - Provider Selection

  /// Currently selected provider
  private(set) var selectedProvider: STTProvider = .whisper

  // MARK: - Configuration Tracking

  /// Loaded Whisper configuration
  private(set) var loadedWhisperModelSize: WhisperModelSize?
  private(set) var loadedWhisperQuantization: WhisperQuantization?

  /// Loaded Fun-ASR configuration
  private(set) var loadedFunASRVariant: FunASRModelVariant?

  // MARK: - State

  /// Whether a model is currently being loaded
  private(set) var isLoading: Bool = false

  /// Model loading progress (0.0 to 1.0)
  private(set) var loadingProgress: Double = 0

  /// Last error that occurred
  private(set) var error: STTError?

  // MARK: - Computed Properties

  var isLoaded: Bool {
    switch selectedProvider {
      case .whisper:
        whisperEngine?.isLoaded ?? false
      case .funASR:
        funASREngine?.isLoaded ?? false
    }
  }

  var isTranscribing: Bool {
    switch selectedProvider {
      case .whisper:
        whisperEngine?.isTranscribing ?? false
      case .funASR:
        funASREngine?.isTranscribing ?? false
    }
  }

  var transcriptionTime: TimeInterval {
    switch selectedProvider {
      case .whisper:
        whisperEngine?.transcriptionTime ?? 0
      case .funASR:
        funASREngine?.transcriptionTime ?? 0
    }
  }

  /// Check if Whisper engine needs reload due to config change
  func whisperNeedsReload(modelSize: WhisperModelSize, quantization: WhisperQuantization) -> Bool {
    loadedWhisperModelSize != modelSize || loadedWhisperQuantization != quantization
  }

  /// Check if Fun-ASR engine needs reload due to config change
  func funASRNeedsReload(variant: FunASRModelVariant) -> Bool {
    loadedFunASRVariant != variant
  }

  // MARK: - Provider Selection

  /// Switch to a different STT provider
  func selectProvider(_ provider: STTProvider) async {
    guard provider != selectedProvider else { return }

    // Stop and unload the previous engine to free GPU memory
    switch selectedProvider {
      case .whisper:
        if let engine = whisperEngine, engine.isLoaded {
          await engine.unload()
        }
      case .funASR:
        if let engine = funASREngine, engine.isLoaded {
          await engine.unload()
        }
    }

    selectedProvider = provider
    error = nil
  }

  // MARK: - Whisper Engine Lifecycle

  /// Load or reload Whisper engine with specified configuration
  func loadWhisperEngine(
    modelSize: WhisperModelSize,
    quantization: WhisperQuantization
  ) async throws {
    // If already loaded with same config, skip
    if whisperEngine?.isLoaded == true,
       loadedWhisperModelSize == modelSize,
       loadedWhisperQuantization == quantization
    {
      return
    }

    // Unload existing engine if config changed
    if whisperEngine != nil {
      await whisperEngine?.unload()
    }

    isLoading = true
    loadingProgress = 0
    error = nil

    MLXMemory.configureForPlatform()

    do {
      whisperEngine = WhisperEngine(modelSize: modelSize, quantization: quantization)

      try await whisperEngine?.load { [weak self] progress in
        Task { @MainActor in
          self?.loadingProgress = progress.fractionCompleted
        }
      }

      loadedWhisperModelSize = modelSize
      loadedWhisperQuantization = quantization
      isLoading = false
      loadingProgress = 1.0
    } catch {
      isLoading = false
      loadingProgress = 0
      whisperEngine = nil
      let sttError = STTError.modelLoadFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  // MARK: - Fun-ASR Engine Lifecycle

  /// Load or reload Fun-ASR engine with specified configuration
  func loadFunASREngine(variant: FunASRModelVariant) async throws {
    // If already loaded with same config, skip
    if funASREngine?.isLoaded == true, loadedFunASRVariant == variant {
      return
    }

    // Unload existing engine if config changed
    if funASREngine != nil {
      await funASREngine?.unload()
    }

    isLoading = true
    loadingProgress = 0
    error = nil

    MLXMemory.configureForPlatform()

    do {
      funASREngine = FunASREngine(variant: variant)

      try await funASREngine?.load { [weak self] progress in
        Task { @MainActor in
          self?.loadingProgress = progress.fractionCompleted
        }
      }

      loadedFunASRVariant = variant
      isLoading = false
      loadingProgress = 1.0
    } catch {
      isLoading = false
      loadingProgress = 0
      funASREngine = nil
      let sttError = STTError.modelLoadFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  // MARK: - Whisper Transcription

  /// Transcribe audio file using Whisper
  func whisperTranscribe(
    url: URL,
    language: Language?,
    timestamps: TimestampGranularity
  ) async throws -> TranscriptionResult {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.transcribe(
        url,
        language: language,
        temperature: 0.0,
        timestamps: timestamps
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Translate audio file to English using Whisper
  func whisperTranslate(
    url: URL,
    language: Language?,
    timestamps: TimestampGranularity
  ) async throws -> TranscriptionResult {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.translate(
        url,
        language: language,
        timestamps: timestamps
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Detect language of audio file using Whisper
  func whisperDetectLanguage(url: URL) async throws -> (Language, Float) {
    guard let engine = whisperEngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.detectLanguage(url)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  // MARK: - Fun-ASR Transcription

  /// Transcribe audio file using Fun-ASR
  func funASRTranscribe(
    url: URL,
    language: FunASRLanguage
  ) async throws -> TranscriptionResult {
    guard let engine = funASREngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.transcribe(
        url,
        language: language,
        temperature: 0.0,
        topP: 0.95,
        topK: 0
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Translate audio file using Fun-ASR
  func funASRTranslate(
    url: URL,
    sourceLanguage: FunASRLanguage,
    targetLanguage: FunASRLanguage
  ) async throws -> TranscriptionResult {
    guard let engine = funASREngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    do {
      return try await engine.translate(
        url,
        sourceLanguage: sourceLanguage,
        targetLanguage: targetLanguage,
        temperature: 0.0,
        topP: 0.95
      )
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      let sttError = (error as? STTError) ?? STTError.transcriptionFailed(underlying: error)
      self.error = sttError
      throw sttError
    }
  }

  /// Stream transcription tokens using Fun-ASR
  func funASRTranscribeStreaming(
    url: URL,
    language: FunASRLanguage
  ) async throws -> AsyncThrowingStream<String, Error> {
    guard let engine = funASREngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    return try await engine.transcribeStreaming(
      url,
      language: language,
      task: .transcribe,
      targetLanguage: .english,
      temperature: 0.0,
      topP: 0.95,
      topK: 0
    )
  }

  /// Stream translation tokens using Fun-ASR
  func funASRTranslateStreaming(
    url: URL,
    sourceLanguage: FunASRLanguage,
    targetLanguage: FunASRLanguage
  ) async throws -> AsyncThrowingStream<String, Error> {
    guard let engine = funASREngine, engine.isLoaded else {
      throw STTError.modelNotLoaded
    }

    error = nil

    return try await engine.transcribeStreaming(
      url,
      language: sourceLanguage,
      task: .translate,
      targetLanguage: targetLanguage,
      temperature: 0.0,
      topP: 0.95,
      topK: 0
    )
  }

  // MARK: - Common Operations

  /// Unload current provider's engine
  func unload() async {
    switch selectedProvider {
      case .whisper:
        await whisperEngine?.unload()
        whisperEngine = nil
        loadedWhisperModelSize = nil
        loadedWhisperQuantization = nil
      case .funASR:
        await funASREngine?.unload()
        funASREngine = nil
        loadedFunASRVariant = nil
    }
  }

  /// Stop current transcription
  func stop() async {
    switch selectedProvider {
      case .whisper:
        await whisperEngine?.stop()
      case .funASR:
        await funASREngine?.stop()
    }
  }
}
