// Copyright © Anthony DePasquale

import AVFoundation
import Foundation
import MLXAudio
import SwiftUI

/// Audio input source
enum AudioSource: String, CaseIterable {
  case file = "File"
  case microphone = "Microphone"
}

/// STT task type
enum STTTask: String, CaseIterable {
  case transcribe = "Transcribe"
  case translate = "Translate"
  case detectLanguage = "Detect Language"

  /// Check if this task is available for the given provider
  func isAvailable(for provider: STTProvider) -> Bool {
    switch self {
      case .transcribe:
        true
      case .translate:
        provider.supportsTranslation
      case .detectLanguage:
        // Only Whisper supports language detection
        provider == .whisper
    }
  }
}

/// Central state management for the STT application
@MainActor
@Observable
final class AppState {
  // MARK: - Dependencies

  let engineManager: EngineManager
  let audioRecorder: AudioRecorder

  // MARK: - Provider Selection

  /// Selected STT provider
  var selectedProvider: STTProvider = .whisper {
    didSet {
      if oldValue != selectedProvider {
        // Switch task if current task not available for new provider
        if !selectedTask.isAvailable(for: selectedProvider) {
          selectedTask = .transcribe
        }
        // Clear results when switching providers
        lastResult = nil
        detectedLanguageResult = nil
        streamingText = ""
      }
    }
  }

  // MARK: - Whisper Configuration

  /// Selected Whisper model size
  var selectedWhisperModelSize: WhisperModelSize = .base

  /// Selected Whisper quantization level
  var selectedWhisperQuantization: WhisperQuantization = .q4

  /// Selected Whisper source language (nil = auto-detect)
  var selectedWhisperLanguage: Language?

  /// Timestamp granularity for Whisper (segment recommended - .none can cause hallucinations)
  var timestampGranularity: TimestampGranularity = .segment

  // MARK: - Fun-ASR Configuration

  /// Selected Fun-ASR model type
  var selectedFunASRModelType: FunASRModelType = .nano

  /// Selected Fun-ASR quantization level
  var selectedFunASRQuantization: FunASRQuantization = .q4

  /// Selected Fun-ASR source language
  var selectedFunASRLanguage: FunASRLanguage = .auto

  /// Selected Fun-ASR target language (for translation)
  var selectedFunASRTargetLanguage: FunASRLanguage = .english

  /// Computed Fun-ASR variant
  var selectedFunASRVariant: FunASRModelVariant {
    FunASRModelVariant(modelType: selectedFunASRModelType, quantization: selectedFunASRQuantization)
  }

  // MARK: - Task Configuration

  /// Selected task (transcribe, translate, detect)
  var selectedTask: STTTask = .transcribe

  // MARK: - Audio Source

  /// Current audio input source
  var audioSource: AudioSource = .file

  /// URL of imported audio file
  var importedFileURL: URL?

  // MARK: - UI State

  /// Whether settings inspector is visible
  var showInspector: Bool = true

  /// Status message to display
  var statusMessage: String = ""

  // MARK: - Results

  /// Last transcription result
  private(set) var lastResult: TranscriptionResult?

  /// Streaming segments during recording (Whisper)
  private(set) var streamingSegments: [TranscriptionSegment] = []

  /// Streaming text during token streaming (Fun-ASR)
  private(set) var streamingText: String = ""

  /// Whether currently streaming tokens (Fun-ASR)
  private(set) var isStreamingTokens: Bool = false

  /// Detected language result (for detect language task)
  private(set) var detectedLanguageResult: (language: Language, confidence: Float)?

  // MARK: - Delegated State (from EngineManager)

  var isLoaded: Bool { engineManager.isLoaded }
  var isTranscribing: Bool { engineManager.isTranscribing || isStreamingTokens }
  var isModelLoading: Bool { engineManager.isLoading }
  var loadingProgress: Double { engineManager.loadingProgress }
  var error: STTError? { engineManager.error }
  var transcriptionTime: TimeInterval { engineManager.transcriptionTime }

  // MARK: - Recording State (from AudioRecorder)

  var isRecording: Bool { audioRecorder.isRecording }
  var recordingDuration: TimeInterval { audioRecorder.duration }
  var recordingURL: URL? { audioRecorder.recordingURL }

  // MARK: - Computed Properties

  var canTranscribe: Bool {
    guard !isTranscribing, !isRecording, !isModelLoading else { return false }
    switch audioSource {
      case .file:
        return importedFileURL != nil
      case .microphone:
        return recordingURL != nil
    }
  }

  var canStartRecording: Bool {
    !isTranscribing && !isRecording && !isModelLoading && audioSource == .microphone
  }

  var needsModelReload: Bool {
    switch selectedProvider {
      case .whisper:
        engineManager.whisperNeedsReload(
          modelSize: selectedWhisperModelSize,
          quantization: selectedWhisperQuantization
        )
      case .funASR:
        engineManager.funASRNeedsReload(variant: selectedFunASRVariant)
    }
  }

  var audioURLToProcess: URL? {
    switch audioSource {
      case .file:
        importedFileURL
      case .microphone:
        recordingURL
    }
  }

  // MARK: - Initialization

  init() {
    engineManager = EngineManager()
    audioRecorder = AudioRecorder()
  }

  // MARK: - Provider Selection

  /// Select a new STT provider
  func selectProvider(_ provider: STTProvider) async {
    guard provider != selectedProvider else { return }
    await engineManager.selectProvider(provider)
    selectedProvider = provider
  }

  // MARK: - Engine Operations

  /// Load the engine with current configuration
  func loadEngine() async throws {
    switch selectedProvider {
      case .whisper:
        try await loadWhisperEngine()
      case .funASR:
        try await loadFunASREngine()
    }
  }

  private func loadWhisperEngine() async throws {
    do {
      try await engineManager.loadWhisperEngine(
        modelSize: selectedWhisperModelSize,
        quantization: selectedWhisperQuantization
      )
      statusMessage =
        "\(selectedWhisperModelSize.displayName) (\(selectedWhisperQuantization.rawValue)) loaded"
    } catch {
      statusMessage = error.localizedDescription
      throw error
    }
  }

  private func loadFunASREngine() async throws {
    do {
      try await engineManager.loadFunASREngine(variant: selectedFunASRVariant)
      statusMessage =
        "\(selectedFunASRModelType.displayName) (\(selectedFunASRQuantization.displayName)) loaded"
    } catch {
      statusMessage = error.localizedDescription
      throw error
    }
  }

  /// Perform transcription/translation/detection based on selected task
  func performTask() async {
    guard let url = audioURLToProcess else {
      statusMessage = "No audio file selected"
      return
    }

    // Load engine if not loaded or config changed
    if !isLoaded || needsModelReload {
      do {
        try await loadEngine()
      } catch {
        return
      }
    }

    statusMessage = "Processing..."
    lastResult = nil
    detectedLanguageResult = nil
    streamingText = ""

    switch selectedProvider {
      case .whisper:
        await performWhisperTask(url: url)
      case .funASR:
        await performFunASRTask(url: url)
    }
  }

  private func performWhisperTask(url: URL) async {
    do {
      switch selectedTask {
        case .transcribe:
          lastResult = try await engineManager.whisperTranscribe(
            url: url,
            language: selectedWhisperLanguage,
            timestamps: timestampGranularity
          )
          if let result = lastResult {
            statusMessage = formatResultStatus(result)
          }

        case .translate:
          lastResult = try await engineManager.whisperTranslate(
            url: url,
            language: selectedWhisperLanguage,
            timestamps: timestampGranularity
          )
          if let result = lastResult {
            statusMessage = formatResultStatus(result)
          }

        case .detectLanguage:
          let (language, confidence) = try await engineManager.whisperDetectLanguage(url: url)
          detectedLanguageResult = (language, confidence)
          let percentage = Int(confidence * 100)
          statusMessage = "Detected: \(language.displayName) (\(percentage)% confidence)"
      }
    } catch is CancellationError {
      statusMessage = "Stopped"
    } catch {
      statusMessage = error.localizedDescription
    }
  }

  private func performFunASRTask(url: URL) async {
    do {
      switch selectedTask {
        case .transcribe:
          await performFunASRStreamingTranscribe(url: url)

        case .translate:
          await performFunASRStreamingTranslate(url: url)

        case .detectLanguage:
          // Fun-ASR doesn't support language detection
          statusMessage = "Language detection not supported for Fun-ASR"
      }
    }
  }

  private func performFunASRStreamingTranscribe(url: URL) async {
    do {
      isStreamingTokens = true
      streamingText = ""
      statusMessage = "Streaming..."

      // Get audio duration for RTF calculation
      let audioDuration = getAudioDuration(url: url)

      let startTime = ContinuousClock.now

      let stream = try await engineManager.funASRTranscribeStreaming(
        url: url,
        language: selectedFunASRLanguage
      )

      for try await token in stream {
        streamingText += token
      }

      let processingTime = ContinuousClock.now - startTime
      isStreamingTokens = false

      // Create final result
      lastResult = TranscriptionResult(
        text: streamingText.trimmingCharacters(in: .whitespacesAndNewlines),
        language: selectedFunASRLanguage.rawValue,
        segments: [],
        processingTime: processingTime.timeInterval,
        duration: audioDuration
      )

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }
    } catch is CancellationError {
      isStreamingTokens = false
      statusMessage = "Stopped"
    } catch {
      isStreamingTokens = false
      statusMessage = error.localizedDescription
    }
  }

  private func performFunASRStreamingTranslate(url: URL) async {
    do {
      isStreamingTokens = true
      streamingText = ""
      statusMessage = "Translating..."

      // Get audio duration for RTF calculation
      let audioDuration = getAudioDuration(url: url)

      let startTime = ContinuousClock.now

      let stream = try await engineManager.funASRTranslateStreaming(
        url: url,
        sourceLanguage: selectedFunASRLanguage,
        targetLanguage: selectedFunASRTargetLanguage
      )

      for try await token in stream {
        streamingText += token
      }

      let processingTime = ContinuousClock.now - startTime
      isStreamingTokens = false

      // Create final result
      lastResult = TranscriptionResult(
        text: streamingText.trimmingCharacters(in: .whitespacesAndNewlines),
        language: selectedFunASRTargetLanguage.rawValue,
        segments: [],
        processingTime: processingTime.timeInterval,
        duration: audioDuration
      )

      if let result = lastResult {
        statusMessage = formatResultStatus(result)
      }
    } catch is CancellationError {
      isStreamingTokens = false
      statusMessage = "Stopped"
    } catch {
      isStreamingTokens = false
      statusMessage = error.localizedDescription
    }
  }

  /// Get the duration of an audio file
  private func getAudioDuration(url: URL) -> TimeInterval {
    do {
      let audioFile = try AVAudioFile(forReading: url)
      let duration = Double(audioFile.length) / audioFile.fileFormat.sampleRate
      return duration
    } catch {
      return 0
    }
  }

  /// Stop current transcription
  func stop() async {
    await engineManager.stop()
    isStreamingTokens = false
    statusMessage = "Stopped"
  }

  // MARK: - Recording Operations

  private var streamingTask: Task<Void, Never>?

  /// Start recording from microphone
  func startRecording() async {
    let hasPermission = await audioRecorder.requestPermission()
    guard hasPermission else {
      statusMessage = "Microphone permission denied"
      return
    }

    // Load model if needed for streaming transcription
    if !isLoaded || needsModelReload {
      do {
        try await loadEngine()
      } catch {
        // Continue without streaming - model will load when transcribing
      }
    }

    do {
      try audioRecorder.startRecording()
      statusMessage = "Recording..."
      lastResult = nil
      streamingSegments = []
      streamingText = ""

      // Start streaming transcription if model is loaded (Whisper only for now)
      if isLoaded, selectedProvider == .whisper {
        startWhisperStreamingTranscription()
      }
    } catch {
      statusMessage = "Failed to start recording: \(error.localizedDescription)"
    }
  }

  /// Stop recording
  func stopRecording() {
    streamingTask?.cancel()
    streamingTask = nil
    audioRecorder.stopRecording()
    statusMessage = "Recording stopped. Ready to transcribe."
  }

  /// Start periodic transcription during recording (Whisper)
  private func startWhisperStreamingTranscription() {
    streamingTask = Task {
      // Wait for initial audio accumulation
      try? await Task.sleep(for: .seconds(3))

      while !Task.isCancelled, isRecording {
        // Transcribe current recording buffer
        if let url = audioRecorder.recordingURL {
          do {
            let result = try await engineManager.whisperTranscribe(
              url: url,
              language: selectedWhisperLanguage,
              timestamps: .segment
            )
            // Update streaming segments
            if !Task.isCancelled {
              streamingSegments = result.segments
              statusMessage = "Recording... (\(Int(recordingDuration))s)"
            }
          } catch {
            // Ignore errors during streaming - file may be in use
          }
        }

        // Wait before next transcription
        try? await Task.sleep(for: .seconds(5))
      }
    }
  }

  /// Append new segments from streaming transcription
  func appendStreamingSegments(_ newSegments: [TranscriptionSegment]) {
    // Merge segments avoiding duplicates based on timing
    for segment in newSegments {
      if !streamingSegments.contains(where: { abs($0.start - segment.start) < 0.5 }) {
        streamingSegments.append(segment)
      }
    }
    streamingSegments.sort { $0.start < $1.start }
  }

  // MARK: - File Import

  /// Set imported file URL
  func setImportedFile(_ url: URL) {
    importedFileURL = url
    statusMessage = "Selected: \(url.lastPathComponent)"
    lastResult = nil
    detectedLanguageResult = nil
    streamingText = ""
  }

  /// Clear imported file
  func clearImportedFile() {
    importedFileURL = nil
    lastResult = nil
    detectedLanguageResult = nil
    streamingText = ""
    statusMessage = ""
  }

  // MARK: - Configuration Changes

  /// Handle Whisper model size change
  func setWhisperModelSize(_ size: WhisperModelSize) {
    guard size != selectedWhisperModelSize else { return }
    selectedWhisperModelSize = size
    lastResult = nil
    detectedLanguageResult = nil
  }

  /// Handle Whisper quantization change
  func setWhisperQuantization(_ quantization: WhisperQuantization) {
    guard quantization != selectedWhisperQuantization else { return }
    selectedWhisperQuantization = quantization
    lastResult = nil
    detectedLanguageResult = nil
  }

  /// Handle Fun-ASR model type change
  func setFunASRModelType(_ modelType: FunASRModelType) {
    guard modelType != selectedFunASRModelType else { return }
    selectedFunASRModelType = modelType
    lastResult = nil
    streamingText = ""
  }

  /// Handle Fun-ASR quantization change
  func setFunASRQuantization(_ quantization: FunASRQuantization) {
    guard quantization != selectedFunASRQuantization else { return }
    selectedFunASRQuantization = quantization
    lastResult = nil
    streamingText = ""
  }

  // MARK: - Private Helpers

  private func formatResultStatus(_ result: TranscriptionResult) -> String {
    let timeStr = String(format: "%.2f", result.processingTime)
    let durationStr = String(format: "%.2f", result.duration)
    let rtfStr = String(format: "%.2f", result.realTimeFactor)
    return "Processed \(durationStr)s audio in \(timeStr)s (RTF: \(rtfStr)x)"
  }
}

// MARK: - Whisper Model Size Display

extension WhisperModelSize {
  var displayName: String {
    switch self {
      case .tiny: "Tiny (39M)"
      case .tinyEn: "Tiny.en (39M)"
      case .base: "Base (74M)"
      case .baseEn: "Base.en (74M)"
      case .small: "Small (244M)"
      case .smallEn: "Small.en (244M)"
      case .medium: "Medium (769M)"
      case .mediumEn: "Medium.en (769M)"
      case .large: "Large-v3 (1.5B)"
      case .largeTurbo: "Large-v3-Turbo (809M)"
    }
  }

  var isEnglishOnly: Bool {
    switch self {
      case .tinyEn, .baseEn, .smallEn, .mediumEn:
        true
      default:
        false
    }
  }
}

// MARK: - Fun-ASR Model Type Display

extension FunASRModelType {
  var displayName: String {
    switch self {
      case .nano: "Nano (Transcription)"
      case .mltNano: "MLT Nano (Multilingual)"
    }
  }
}

// MARK: - Fun-ASR Quantization Display

extension FunASRQuantization {
  var displayName: String {
    switch self {
      case .q4: "4-bit (Fastest)"
      case .q8: "8-bit (Balanced)"
      case .fp16: "FP16 (Best Quality)"
    }
  }
}

// MARK: - Duration Extension

extension Duration {
  /// Convert Duration to TimeInterval (seconds)
  var timeInterval: TimeInterval {
    let (seconds, attoseconds) = components
    return Double(seconds) + Double(attoseconds) / 1e18
  }
}
