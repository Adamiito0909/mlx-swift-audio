// Copyright © Anthony DePasquale

import Foundation

/// Core protocol that all STT engines must conform to.
///
/// Transcription configuration and output are engine-specific since each engine
/// may have different capabilities (multilingual, translation, timestamps, etc.)
@MainActor
public protocol STTEngine: Observable {
  /// The provider type for this engine
  var provider: STTProvider { get }

  // MARK: - State Properties

  /// Whether the model is loaded and ready for transcription
  var isLoaded: Bool { get }

  /// Whether transcription is currently in progress
  var isTranscribing: Bool { get }

  /// Time taken for the last transcription (seconds)
  var transcriptionTime: TimeInterval { get }

  // MARK: - Lifecycle Methods

  /// Load the model with optional progress reporting
  /// - Parameter progressHandler: Optional callback for download/load progress
  func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws

  /// Stop any ongoing transcription
  func stop() async

  /// Unload model weights to free GPU memory.
  ///
  /// Preserves cached data (tokenizer, audio buffers) for faster reload.
  /// Use this when switching between engines to free memory while keeping
  /// expensive pre-computed data.
  func unload() async

  /// Full cleanup - releases everything including cached data.
  ///
  /// Use before deallocating the engine or when you need to free all resources.
  func cleanup() async throws
}

// MARK: - Default Implementations

public extension STTEngine {
  /// Load the model without progress reporting
  func load() async throws {
    try await load(progressHandler: nil)
  }
}

// MARK: - Factory

/// Namespace for discovering and creating STT engines with full type safety.
///
/// Each method returns a concrete engine type, enabling autocomplete for
/// engine-specific features.
///
/// ```swift
/// // Default 4-bit quantized (smallest, fastest)
/// let engine = STT.whisper(model: .base)
///
/// // 8-bit quantized (balanced quality and size)
/// let engine = STT.whisper(model: .large, quantization: .q8)
///
/// // Full precision (best quality, larger size)
/// let engine = STT.whisper(model: .largeTurbo, quantization: .fp16)
/// ```
@MainActor
public enum STT {
  /// Whisper: multilingual speech recognition
  ///
  /// - Parameters:
  ///   - model: Model size (tiny, base, small, medium, large, largeTurbo, or English-only variants)
  ///   - quantization: Quantization level (fp16, q8, q4). Default is q4.
  /// - Returns: Configured WhisperEngine instance
  public static func whisper(
    model: WhisperModelSize = .base,
    quantization: WhisperQuantization = .q4
  ) -> WhisperEngine {
    WhisperEngine(modelSize: model, quantization: quantization)
  }

  /// Fun-ASR: LLM-based multilingual speech recognition
  ///
  /// Combines SenseVoice encoder with Qwen3 decoder for high-quality
  /// transcription and translation.
  ///
  /// - Parameter variant: Model variant specification. Default is nano4bit.
  /// - Returns: Configured FunASREngine instance
  public static func funASR(
    variant: FunASRModelVariant = .nano4bit
  ) -> FunASREngine {
    FunASREngine(variant: variant)
  }

  /// Fun-ASR: LLM-based multilingual speech recognition
  ///
  /// Combines SenseVoice encoder with Qwen3 decoder for high-quality
  /// transcription and translation.
  ///
  /// - Parameters:
  ///   - modelType: Model type (.nano for transcription, .mltNano for translation)
  ///   - quantization: Quantization level (.q4, .q8, or .fp16)
  /// - Returns: Configured FunASREngine instance
  public static func funASR(
    modelType: FunASRModelType = .nano,
    quantization: FunASRQuantization = .q4
  ) -> FunASREngine {
    FunASREngine(modelType: modelType, quantization: quantization)
  }
}
