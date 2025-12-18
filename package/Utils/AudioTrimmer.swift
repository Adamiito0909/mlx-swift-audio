// Copyright © Anthony DePasquale
//
// Audio trimming utilities for TTS reference audio preprocessing
//
// Provides:
// 1. Energy-based silence trimming (like librosa.effects.trim)
// 2. Smart word-boundary clipping using Whisper transcription

import Foundation
import MLX

// MARK: - Configuration

/// Configuration for audio trimming operations
public struct AudioTrimConfig: Sendable {
  /// Threshold in dB below peak to consider as silence
  /// Default: 60dB (matching CosyVoice2 Python implementation)
  public var topDb: Float

  /// Frame length in seconds for energy calculation
  /// Default: 25ms (matching CosyVoice2 Python implementation)
  public var frameLength: Float

  /// Hop length in seconds between frames
  /// Default: 12.5ms (matching CosyVoice2 Python implementation)
  public var hopLength: Float

  /// Number of trailing words to drop as safety margin
  /// Helps avoid hallucinated words at audio boundaries
  /// Default: 1
  public var trailingWordsToDrop: Int

  /// Minimum word probability to consider reliable
  /// Words with lower probability may be hallucinations
  /// Default: 0.3
  public var minWordProbability: Float

  /// Maximum word duration anomaly threshold
  /// Words longer than this (in seconds) are considered suspicious
  /// Default: 2.0
  public var maxWordDuration: Float

  /// Presets for different TTS models
  public static let cosyVoice2 = AudioTrimConfig(
    topDb: 60.0,
    frameLength: 0.025,
    hopLength: 0.0125,
    trailingWordsToDrop: 1,
    minWordProbability: 0.3,
    maxWordDuration: 2.0
  )

  public static let chatterbox = AudioTrimConfig(
    topDb: 20.0, // Chatterbox uses more aggressive trimming
    frameLength: 0.025,
    hopLength: 0.0125,
    trailingWordsToDrop: 1,
    minWordProbability: 0.3,
    maxWordDuration: 2.0
  )

  public static let `default` = AudioTrimConfig()

  public init(
    topDb: Float = 60.0,
    frameLength: Float = 0.025,
    hopLength: Float = 0.0125,
    trailingWordsToDrop: Int = 1,
    minWordProbability: Float = 0.3,
    maxWordDuration: Float = 2.0
  ) {
    self.topDb = topDb
    self.frameLength = frameLength
    self.hopLength = hopLength
    self.trailingWordsToDrop = trailingWordsToDrop
    self.minWordProbability = minWordProbability
    self.maxWordDuration = maxWordDuration
  }
}

// MARK: - Trim Result

/// Result of smart audio trimming with optional transcription
public struct AudioTrimResult: Sendable {
  /// The trimmed audio samples
  public let audio: [Float]

  /// Sample rate of the audio
  public let sampleRate: Int

  /// The transcription of the trimmed audio (if Whisper was used)
  public let transcription: String?

  /// Word timings for the trimmed audio (if available)
  public let words: [Word]?

  /// Original duration before trimming (seconds)
  public let originalDuration: Float

  /// Final duration after trimming (seconds)
  public let trimmedDuration: Float

  /// Whether the audio was clipped at a word boundary
  public let clippedAtWordBoundary: Bool

  public init(
    audio: [Float],
    sampleRate: Int,
    transcription: String? = nil,
    words: [Word]? = nil,
    originalDuration: Float,
    trimmedDuration: Float,
    clippedAtWordBoundary: Bool = false
  ) {
    self.audio = audio
    self.sampleRate = sampleRate
    self.transcription = transcription
    self.words = words
    self.originalDuration = originalDuration
    self.trimmedDuration = trimmedDuration
    self.clippedAtWordBoundary = clippedAtWordBoundary
  }
}

// MARK: - AudioTrimmer

/// Utilities for trimming silence and clipping audio at word boundaries
public enum AudioTrimmer {
  // MARK: - Energy-based Silence Trimming

  /// Trim silence from beginning and end of audio
  ///
  /// This is equivalent to `librosa.effects.trim` in Python.
  /// It calculates RMS energy per frame and removes frames below a threshold
  /// relative to the peak energy.
  ///
  /// - Parameters:
  ///   - audio: Audio samples (mono)
  ///   - sampleRate: Sample rate in Hz
  ///   - config: Trimming configuration (default: `.default`)
  /// - Returns: Trimmed audio samples, or original if no silence detected
  public static func trimSilence(
    _ audio: [Float],
    sampleRate: Int,
    config: AudioTrimConfig = .default
  ) -> [Float] {
    guard !audio.isEmpty else { return audio }

    let frameSamples = Int(config.frameLength * Float(sampleRate))
    let hopSamples = Int(config.hopLength * Float(sampleRate))

    guard frameSamples > 0, hopSamples > 0 else { return audio }

    // Calculate number of frames
    let numFrames = max(1, (audio.count - frameSamples) / hopSamples + 1)

    // Calculate RMS energy per frame in dB
    var rmsDb = [Float](repeating: -Float.infinity, count: numFrames)

    for i in 0 ..< numFrames {
      let start = i * hopSamples
      let end = min(start + frameSamples, audio.count)

      // Calculate RMS energy
      var sumSquares: Float = 0
      for j in start ..< end {
        sumSquares += audio[j] * audio[j]
      }
      let rms = (sumSquares / Float(end - start)).squareRoot()

      // Convert to dB (with floor to avoid -inf)
      rmsDb[i] = 20 * log10(max(rms, 1e-10))
    }

    // Find peak dB and threshold
    guard let maxDb = rmsDb.max(), maxDb > -Float.infinity else { return audio }
    let threshold = maxDb - config.topDb

    // Find first frame above threshold (start of speech)
    var startFrame = 0
    for i in 0 ..< numFrames {
      if rmsDb[i] >= threshold {
        startFrame = i
        break
      }
    }

    // Find last frame above threshold (end of speech)
    var endFrame = numFrames - 1
    for i in stride(from: numFrames - 1, through: 0, by: -1) {
      if rmsDb[i] >= threshold {
        endFrame = i
        break
      }
    }

    // Convert frames to samples
    // Start: first sample of first non-silent frame
    let startSample = startFrame * hopSamples
    // End: matches librosa's frames_to_samples(nonzero[-1] + 1, hop_length)
    // This is the start of the next frame after the last non-silent frame
    let endSample = min((endFrame + 1) * hopSamples, audio.count)

    guard startSample < endSample else { return audio }

    return Array(audio[startSample ..< endSample])
  }

  /// Trim silence from beginning and end of audio
  ///
  /// Convenience overload that returns sample indices instead of trimmed audio.
  ///
  /// - Parameters:
  ///   - audio: Audio samples (mono)
  ///   - sampleRate: Sample rate in Hz
  ///   - config: Trimming configuration
  /// - Returns: Tuple of (startSample, endSample) indices, or nil if no speech detected
  public static func findSpeechBounds(
    _ audio: [Float],
    sampleRate: Int,
    config: AudioTrimConfig = .default
  ) -> (start: Int, end: Int)? {
    guard !audio.isEmpty else { return nil }

    let frameSamples = Int(config.frameLength * Float(sampleRate))
    let hopSamples = Int(config.hopLength * Float(sampleRate))

    guard frameSamples > 0, hopSamples > 0 else { return nil }

    let numFrames = max(1, (audio.count - frameSamples) / hopSamples + 1)
    var rmsDb = [Float](repeating: -Float.infinity, count: numFrames)

    for i in 0 ..< numFrames {
      let start = i * hopSamples
      let end = min(start + frameSamples, audio.count)

      var sumSquares: Float = 0
      for j in start ..< end {
        sumSquares += audio[j] * audio[j]
      }
      let rms = (sumSquares / Float(end - start)).squareRoot()
      rmsDb[i] = 20 * log10(max(rms, 1e-10))
    }

    guard let maxDb = rmsDb.max(), maxDb > -Float.infinity else { return nil }
    let threshold = maxDb - config.topDb

    var startFrame: Int?
    var endFrame: Int?

    for i in 0 ..< numFrames {
      if rmsDb[i] >= threshold {
        startFrame = i
        break
      }
    }

    for i in stride(from: numFrames - 1, through: 0, by: -1) {
      if rmsDb[i] >= threshold {
        endFrame = i
        break
      }
    }

    guard let start = startFrame, let end = endFrame else { return nil }

    // Convert frames to samples (matching librosa's behavior)
    let startSample = start * hopSamples
    let endSample = min((end + 1) * hopSamples, audio.count)

    guard startSample < endSample else { return nil }

    return (startSample, endSample)
  }

  // MARK: - Word Boundary Utilities

  /// Calculate anomaly score for a word
  ///
  /// Higher scores indicate the word is likely a hallucination or timing error.
  /// Based on WhisperTiming.swift heuristics.
  ///
  /// - Parameter word: The word to score
  /// - Returns: Anomaly score (0 = normal, higher = more anomalous)
  public static func wordAnomalyScore(_ word: Word) -> Float {
    var score: Float = 0

    // Low probability indicates uncertainty
    if word.probability < 0.15 {
      score += 1.0
    }

    let duration = Float(word.end - word.start)

    // Very short duration (< 133ms) is suspicious
    if duration < 0.133 {
      score += (0.133 - duration) * 15
    }

    // Very long duration (> 2s) is suspicious
    if duration > 2.0 {
      score += duration - 2.0
    }

    return score
  }

  /// Filter out unreliable trailing words that may be hallucinations
  ///
  /// This is critical for reference audio that may have been clipped mid-word.
  /// Whisper can hallucinate complete words from partial audio at the end.
  ///
  /// - Parameters:
  ///   - words: Array of words from transcription
  ///   - audioDuration: Actual duration of the audio in seconds
  ///   - config: Trimming configuration
  /// - Returns: Filtered array with unreliable trailing words removed
  public static func dropUnreliableTrailingWords(
    _ words: [Word],
    audioDuration: Float,
    config: AudioTrimConfig = .default
  ) -> [Word] {
    var result = words

    // 1. Drop words that claim to end after actual audio duration
    //    (clear sign of hallucination - Whisper invented audio that doesn't exist)
    while let last = result.last, Float(last.end) > audioDuration + 0.05 {
      Log.tts.debug("Dropping word '\(last.word)' - ends at \(last.end)s but audio is only \(audioDuration)s")
      result.removeLast()
    }

    // 2. Drop words with low probability or high anomaly score
    while let last = result.last {
      let anomaly = wordAnomalyScore(last)
      if anomaly > 0.5 || last.probability < config.minWordProbability {
        Log.tts.debug("Dropping word '\(last.word)' - anomaly=\(anomaly), prob=\(last.probability)")
        result.removeLast()
      } else {
        break
      }
    }

    // 3. Drop additional words as safety margin
    //    Even "good" words at the boundary may be affected by audio cutoff
    for _ in 0 ..< config.trailingWordsToDrop {
      if result.count > 1 {
        if let last = result.last {
          Log.tts.debug("Dropping word '\(last.word)' as safety margin")
        }
        result.removeLast()
      }
    }

    return result
  }

  /// Find the optimal clip point based on word boundaries
  ///
  /// Given a maximum duration, finds the last reliable word that ends before
  /// that duration and returns the sample index to clip at.
  ///
  /// - Parameters:
  ///   - words: Array of words with timestamps
  ///   - maxDuration: Maximum allowed duration in seconds
  ///   - sampleRate: Sample rate in Hz
  ///   - safetyMargin: Additional margin in seconds (default: 0.1s)
  /// - Returns: Tuple of (clipSampleIndex, validWords) or nil if no valid words
  public static func findWordBoundaryClipPoint(
    words: [Word],
    maxDuration: Float,
    sampleRate: Int,
    safetyMargin: Float = 0.1
  ) -> (clipSample: Int, words: [Word])? {
    let targetDuration = maxDuration - safetyMargin

    // Find words that end before target duration
    let validWords = words.filter { Float($0.end) <= targetDuration }

    guard let lastWord = validWords.last else {
      // If no words fit, try to use just the first word
      if let first = words.first, Float(first.end) <= maxDuration {
        let clipSample = Int(Float(first.end) * Float(sampleRate))
        return (clipSample, [first])
      }
      return nil
    }

    let clipSample = Int(Float(lastWord.end) * Float(sampleRate))
    return (clipSample, validWords)
  }

  // MARK: - Combined Trimming

  /// Trim audio with silence removal only
  ///
  /// Use this when you don't need word-level accuracy or don't have Whisper available.
  ///
  /// - Parameters:
  ///   - audio: Audio samples (mono)
  ///   - sampleRate: Sample rate in Hz
  ///   - maxDuration: Optional maximum duration in seconds
  ///   - config: Trimming configuration
  /// - Returns: AudioTrimResult with trimmed audio
  public static func trim(
    audio: [Float],
    sampleRate: Int,
    maxDuration: Float? = nil,
    config: AudioTrimConfig = .default
  ) -> AudioTrimResult {
    let originalDuration = Float(audio.count) / Float(sampleRate)

    // Step 1: Trim silence
    var trimmed = trimSilence(audio, sampleRate: sampleRate, config: config)

    // Step 2: Apply max duration if specified
    if let maxDuration {
      let maxSamples = Int(maxDuration * Float(sampleRate))
      if trimmed.count > maxSamples {
        trimmed = Array(trimmed.prefix(maxSamples))
      }
    }

    let trimmedDuration = Float(trimmed.count) / Float(sampleRate)

    return AudioTrimResult(
      audio: trimmed,
      sampleRate: sampleRate,
      transcription: nil,
      words: nil,
      originalDuration: originalDuration,
      trimmedDuration: trimmedDuration,
      clippedAtWordBoundary: false
    )
  }

  /// Trim audio at word boundaries using Whisper transcription
  ///
  /// This provides the highest quality trimming by:
  /// 1. Trimming silence from beginning and end
  /// 2. Transcribing the audio with word-level timestamps
  /// 3. Clipping at the last reliable word boundary before maxDuration
  ///
  /// This prevents audio from being cut mid-word, which can cause TTS artifacts.
  ///
  /// - Parameters:
  ///   - audio: Audio samples (mono) at 16kHz (required for Whisper)
  ///   - sampleRate: Sample rate in Hz (should be 16000 for Whisper)
  ///   - maxDuration: Maximum allowed duration in seconds
  ///   - whisperEngine: WhisperEngine instance for transcription
  ///   - config: Trimming configuration
  /// - Returns: AudioTrimResult with trimmed audio, transcription, and word timings
  @MainActor
  public static func trimAtWordBoundary(
    audio: [Float],
    sampleRate: Int,
    maxDuration: Float,
    whisperEngine: WhisperEngine,
    config: AudioTrimConfig = .default
  ) async throws -> AudioTrimResult {
    let originalDuration = Float(audio.count) / Float(sampleRate)

    // Step 1: Trim silence first
    let silenceTrimmed = trimSilence(audio, sampleRate: sampleRate, config: config)
    let silenceTrimmedDuration = Float(silenceTrimmed.count) / Float(sampleRate)

    Log.tts.debug("Silence trimming: \(originalDuration)s -> \(silenceTrimmedDuration)s")

    // If already under max duration after silence trim, we're done
    if silenceTrimmedDuration <= maxDuration {
      return AudioTrimResult(
        audio: silenceTrimmed,
        sampleRate: sampleRate,
        transcription: nil,
        words: nil,
        originalDuration: originalDuration,
        trimmedDuration: silenceTrimmedDuration,
        clippedAtWordBoundary: false
      )
    }

    // Step 2: Transcribe with word timestamps
    // Note: Whisper expects 16kHz audio
    let audioForWhisper: MLXArray = if sampleRate != 16000 {
      // Resample to 16kHz for Whisper
      AudioResampler.resample(
        MLXArray(silenceTrimmed),
        from: sampleRate,
        to: 16000
      )
    } else {
      MLXArray(silenceTrimmed)
    }

    let result = try await whisperEngine.transcribe(
      audioForWhisper,
      timestamps: .word
    )

    // Collect all words from all segments
    var allWords = result.segments.flatMap { $0.words ?? [] }

    guard !allWords.isEmpty else {
      // No words detected - fall back to simple truncation
      Log.tts.warning("No words detected in transcription, falling back to simple truncation")
      let maxSamples = Int(maxDuration * Float(sampleRate))
      let truncated = Array(silenceTrimmed.prefix(maxSamples))
      return AudioTrimResult(
        audio: truncated,
        sampleRate: sampleRate,
        transcription: result.text,
        words: nil,
        originalDuration: originalDuration,
        trimmedDuration: maxDuration,
        clippedAtWordBoundary: false
      )
    }

    Log.tts.debug("Transcribed \(allWords.count) words")

    // Step 3: Drop unreliable trailing words
    allWords = dropUnreliableTrailingWords(
      allWords,
      audioDuration: silenceTrimmedDuration,
      config: config
    )

    Log.tts.debug("After dropping unreliable words: \(allWords.count) words")

    // Step 4: Find word boundary clip point
    guard let (clipSample, validWords) = findWordBoundaryClipPoint(
      words: allWords,
      maxDuration: maxDuration,
      sampleRate: sampleRate
    ) else {
      // No valid words found - fall back to simple truncation
      Log.tts.warning("No valid words found for clipping, falling back to simple truncation")
      let maxSamples = Int(maxDuration * Float(sampleRate))
      let truncated = Array(silenceTrimmed.prefix(maxSamples))
      return AudioTrimResult(
        audio: truncated,
        sampleRate: sampleRate,
        transcription: result.text,
        words: allWords,
        originalDuration: originalDuration,
        trimmedDuration: maxDuration,
        clippedAtWordBoundary: false
      )
    }

    // Step 5: Clip at word boundary
    let clippedAudio = Array(silenceTrimmed.prefix(clipSample))
    let transcription = validWords.map(\.word).joined()
    let trimmedDuration = Float(clippedAudio.count) / Float(sampleRate)

    Log.tts.debug("Clipped at word boundary: \(silenceTrimmedDuration)s -> \(trimmedDuration)s")
    Log.tts.debug("Transcription: \(transcription)")

    return AudioTrimResult(
      audio: clippedAudio,
      sampleRate: sampleRate,
      transcription: transcription,
      words: validWords,
      originalDuration: originalDuration,
      trimmedDuration: trimmedDuration,
      clippedAtWordBoundary: true
    )
  }
}
