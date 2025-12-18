// Copyright © Anthony DePasquale

import Foundation

// MARK: - TranscriptionResult

/// Complete transcription result from STT engine
public struct TranscriptionResult: Sendable {
  /// Full transcription text
  public let text: String

  /// Detected or specified language code (e.g., "en", "zh", "es")
  public let language: String

  /// Individual segments with timestamps
  public let segments: [TranscriptionSegment]

  /// Processing time in seconds
  public let processingTime: TimeInterval

  /// Audio duration in seconds
  public let duration: TimeInterval

  /// Real-time factor (processingTime / duration)
  /// Values < 1.0 mean faster than real-time
  public var realTimeFactor: Double {
    duration > 0 ? processingTime / duration : 0
  }

  public init(
    text: String,
    language: String,
    segments: [TranscriptionSegment],
    processingTime: TimeInterval,
    duration: TimeInterval
  ) {
    self.text = text
    self.language = language
    self.segments = segments
    self.processingTime = processingTime
    self.duration = duration
  }
}

// MARK: - Segment

/// A transcription segment with timestamps
public struct TranscriptionSegment: Sendable {
  /// Segment text
  public let text: String

  /// Start time in seconds
  public let start: TimeInterval

  /// End time in seconds
  public let end: TimeInterval

  /// Token IDs for this segment
  public let tokens: [Int]

  /// Average log probability
  public let avgLogProb: Float

  /// No-speech probability (0-1)
  /// Higher values indicate the segment likely contains no speech
  public let noSpeechProb: Float

  /// Word-level timestamps (optional, available when using TimestampGranularity.word)
  public let words: [Word]?

  public init(
    text: String,
    start: TimeInterval,
    end: TimeInterval,
    tokens: [Int],
    avgLogProb: Float,
    noSpeechProb: Float,
    words: [Word]? = nil
  ) {
    self.text = text
    self.start = start
    self.end = end
    self.tokens = tokens
    self.avgLogProb = avgLogProb
    self.noSpeechProb = noSpeechProb
    self.words = words
  }
}

// MARK: - Word

/// Word-level timestamp extracted using cross-attention alignment and DTW
public struct Word: Sendable {
  /// The word text
  public let word: String

  /// Start time in seconds
  public let start: TimeInterval

  /// End time in seconds
  public let end: TimeInterval

  /// Confidence probability (0-1)
  public let probability: Float

  public init(
    word: String,
    start: TimeInterval,
    end: TimeInterval,
    probability: Float
  ) {
    self.word = word
    self.start = start
    self.end = end
    self.probability = probability
  }
}

// MARK: - TranscriptionTask

/// The task to perform during transcription
public enum TranscriptionTask: String, Sendable, CaseIterable {
  /// Transcribe in the original language
  case transcribe

  /// Translate to English
  case translate

  public var displayName: String {
    switch self {
      case .transcribe: "Transcribe"
      case .translate: "Translate to English"
    }
  }
}

// MARK: - TimestampGranularity

/// Granularity of timestamps in transcription
public enum TimestampGranularity: Sendable {
  /// No timestamps
  case none

  /// Segment-level timestamps only
  case segment

  /// Word-level timestamps using cross-attention alignment and DTW
  case word

  public var displayName: String {
    switch self {
      case .none: "None"
      case .segment: "Segment-level"
      case .word: "Word-level"
    }
  }
}

// MARK: - WhisperQuantization

/// Quantization level for Whisper models
public enum WhisperQuantization: String, Sendable, CaseIterable {
  /// 16-bit floating point (best quality, larger size)
  case fp16

  /// 8-bit quantization (good balance of quality and size)
  case q8 = "8bit"

  /// 4-bit quantization (smallest size, some quality tradeoff)
  case q4 = "4bit"

  /// Display name
  public var displayName: String {
    switch self {
      case .fp16: "FP16 (Best Quality)"
      case .q8: "8-bit (Balanced)"
      case .q4: "4-bit (Smallest)"
    }
  }

  /// Approximate size multiplier relative to fp16
  public var sizeMultiplier: Float {
    switch self {
      case .fp16: 1.0
      case .q8: 0.5
      case .q4: 0.25
    }
  }
}

// MARK: - WhisperModelSize

/// Available Whisper model sizes
public enum WhisperModelSize: String, Sendable, CaseIterable {
  // Multilingual models (use multilingual.tiktoken)
  case tiny = "whisper-tiny"
  case base = "whisper-base"
  case small = "whisper-small"
  case medium = "whisper-medium"
  case large = "whisper-large-v3"
  case largeTurbo = "whisper-large-v3-turbo"

  // English-only models (use gpt2.tiktoken)
  case tinyEn = "whisper-tiny.en"
  case baseEn = "whisper-base.en"
  case smallEn = "whisper-small.en"
  case mediumEn = "whisper-medium.en"

  /// Whether this model is currently available
  ///
  /// All models are now available via mlx-community repos with proper safetensors format,
  /// converted using mlx-audio-plus.
  public var isAvailable: Bool {
    true
  }

  /// HuggingFace repository ID with specified quantization
  ///
  /// All models use mlx-community repos, converted using mlx-audio-plus
  /// with proper safetensors format.
  ///
  /// - Parameter quantization: The quantization level (default: q4)
  /// - Returns: The HuggingFace repository ID
  public func repoId(quantization: WhisperQuantization = .q4) -> String {
    "mlx-community/\(rawValue)-\(quantization.rawValue)"
  }

  /// HuggingFace repository ID (fp16 precision)
  ///
  /// Convenience property that returns the fp16 variant.
  /// Use `repoId(quantization:)` for other quantization levels.
  public var repoId: String {
    repoId(quantization: .fp16)
  }

  /// Approximate parameter count
  public var parameters: String {
    switch self {
      case .tiny, .tinyEn: "39M"
      case .base, .baseEn: "74M"
      case .small, .smallEn: "244M"
      case .medium, .mediumEn: "769M"
      case .large: "1550M"
      case .largeTurbo: "809M"
    }
  }

  /// Display name
  public var displayName: String {
    switch self {
      case .tiny: "Tiny (39M)"
      case .tinyEn: "Tiny English-only (39M)"
      case .base: "Base (74M)"
      case .baseEn: "Base English-only (74M)"
      case .small: "Small (244M)"
      case .smallEn: "Small English-only (244M)"
      case .medium: "Medium (769M)"
      case .mediumEn: "Medium English-only (769M)"
      case .large: "Large v3 (1550M)"
      case .largeTurbo: "Large v3 Turbo (809M)"
    }
  }
}
