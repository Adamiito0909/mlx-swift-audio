// Copyright ® Canopy Labs (original model implementation)
// Ported to MLX from https://github.com/canopyai/Orpheus-TTS
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/orpheus.txt

import Foundation

/// Orpheus TTS engine - high quality with emotional expressions
///
/// Supports expressions: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`,
/// `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
@Observable
@MainActor
public final class OrpheusEngine: TTSEngine {
  // MARK: - Voice

  /// Available voices for Orpheus TTS
  public enum Voice: String, CaseIterable, Sendable {
    case tara // Female, conversational, clear
    case leah // Female, warm, gentle
    case jess // Female, energetic, youthful
    case leo // Male, authoritative, deep
    case dan // Male, friendly, casual
    case mia // Female, professional, articulate
    case zac // Male, enthusiastic, dynamic
    case zoe // Female, calm, soothing

    /// Convert to generic Voice struct for UI display
    public func toVoice() -> MLXAudio.Voice {
      MLXAudio.Voice(id: rawValue, displayName: rawValue.capitalized, languageCode: "en-US")
    }

    /// All voices as generic Voice structs
    public static var allVoices: [MLXAudio.Voice] {
      allCases.map { $0.toVoice() }
    }
  }

  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .orpheus
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Orpheus-Specific Properties

  /// Temperature for sampling (higher = more variation)
  public var temperature: Float = 0.6

  /// Top-p (nucleus) sampling threshold
  public var topP: Float = 0.8

  // MARK: - Private Properties

  @ObservationIgnored private var orpheusTTS: OrpheusTTS?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: TTSProvider.orpheus.sampleRate)

  // MARK: - Initialization

  public init() {
    Log.tts.debug("OrpheusEngine initialized")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("OrpheusEngine already loaded")
      return
    }

    Log.model.info("Loading Orpheus TTS model...")

    do {
      orpheusTTS = try await OrpheusTTS.load(
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Orpheus TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Orpheus model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 },
    )
    Log.tts.debug("OrpheusEngine stopped")
  }

  public func unload() async {
    await stop()
    orpheusTTS = nil
    isLoaded = false
    Log.tts.debug("OrpheusEngine unloaded")
  }

  public func cleanup() async throws {
    await unload()
  }

  // MARK: - Playback

  public func play(_ audio: AudioResult) async {
    await playback.play(audio, setPlaying: { self.isPlaying = $0 })
  }

  // MARK: - Generation

  /// Generate audio from text
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: Voice,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.collectStream(
      generateStreaming(text, voice: voice),
    )

    Log.tts.timing("Orpheus generation", duration: processingTime)
    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }

  /// Generate and immediately play audio
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  public func say(
    _ text: String,
    voice: Voice,
  ) async throws {
    let audio = try await generate(text, voice: voice)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks
  ///
  /// Text splitting is handled by OrpheusTTS.
  ///
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    voice: Voice,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = provider.sampleRate
    let temp = temperature
    let top = topP

    return playback.createGenerationStream(
      setGenerating: { self.isGenerating = $0 },
      setGenerationTime: { self.generationTime = $0 },
    ) { [weak self] in
      guard let self else {
        return AsyncThrowingStream { $0.finish() }
      }

      // Auto-load if needed
      if !isLoaded {
        try await load()
      }

      guard let orpheusTTS else {
        throw TTSError.modelNotLoaded
      }

      // Wrap model stream to convert [Float] -> AudioChunk
      let modelStream = await orpheusTTS.generateStreaming(
        text: trimmedText,
        voice: voice,
        temperature: temp,
        topP: top,
      )

      let startTime = Date()
      return AsyncThrowingStream { continuation in
        Task {
          do {
            for try await samples in modelStream {
              guard !Task.isCancelled else { break }
              let chunk = AudioChunk(
                samples: samples,
                sampleRate: sampleRate,
                processingTime: Date().timeIntervalSince(startTime),
              )
              continuation.yield(chunk)
            }
            continuation.finish()
          } catch is CancellationError {
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
      }
    }
  }

  /// Play audio with streaming (plays as chunks arrive)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  @discardableResult
  public func sayStreaming(
    _ text: String,
    voice: Voice,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(text, voice: voice),
      setPlaying: { self.isPlaying = $0 },
    )

    lastGeneratedAudioURL = playback.saveAudioFile(samples: samples, sampleRate: provider.sampleRate)

    return .samples(
      data: samples,
      sampleRate: provider.sampleRate,
      processingTime: processingTime,
    )
  }
}
