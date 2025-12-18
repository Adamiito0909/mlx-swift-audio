// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import AVFoundation
import Foundation
import MLXAudio

/// Kokoro TTS engine - fast, lightweight TTS with many voice options
@Observable
@MainActor
public final class KokoroEngine: TTSEngine {
  // MARK: - Voice

  /// Available voices for Kokoro TTS
  public enum Voice: String, CaseIterable, Sendable {
    // American Female
    case afAlloy
    case afAoede
    case afBella
    case afHeart
    case afJessica
    case afKore
    case afNicole
    case afNova
    case afRiver
    case afSarah
    case afSky
    // American Male
    case amAdam
    case amEcho
    case amEric
    case amFenrir
    case amLiam
    case amMichael
    case amOnyx
    case amPuck
    case amSanta
    // British Female
    case bfAlice
    case bfEmma
    case bfIsabella
    case bfLily
    // British Male
    case bmDaniel
    case bmFable
    case bmGeorge
    case bmLewis
    // Spanish
    case efDora
    case emAlex
    // French
    case ffSiwis
    // Hindi
    case hfAlpha
    case hfBeta
    case hfOmega
    case hmPsi
    // Italian
    case ifSara
    case imNicola
    // Japanese
    case jfAlpha
    case jfGongitsune
    case jfNezumi
    case jfTebukuro
    case jmKumo
    // Portuguese
    case pfDora
    case pmSanta
    // Chinese
    case zfXiaobei
    case zfXiaoni
    case zfXiaoxiao
    case zfXiaoyi
    case zmYunjian
    case zmYunxi
    case zmYunxia
    case zmYunyang

    /// Voice ID in snake_case format (e.g., "af_heart")
    public var voiceID: String {
      // Convert camelCase rawValue to snake_case (e.g., "afHeart" -> "af_heart")
      let raw = rawValue
      guard raw.count >= 2 else { return raw.lowercased() }
      // Insert underscore after the 2-letter prefix
      let prefix = raw.prefix(2).lowercased()
      let name = raw.dropFirst(2).lowercased()
      return "\(prefix)_\(name)"
    }

    /// Convert to generic Voice struct for UI display
    public func toVoice() -> MLXAudio.Voice {
      MLXAudio.Voice.fromKokoroID(voiceID)
    }

    /// All voices as generic Voice structs
    public static var allVoices: [MLXAudio.Voice] {
      allCases.map { $0.toVoice() }
    }
  }

  // MARK: - TTSEngine Protocol Properties

  public let provider: TTSProvider = .kokoro
  public let streamingGranularity: StreamingGranularity = .sentence
  public private(set) var isLoaded: Bool = false
  public private(set) var isGenerating: Bool = false
  public private(set) var isPlaying: Bool = false
  public private(set) var lastGeneratedAudioURL: URL?
  public private(set) var generationTime: TimeInterval = 0

  // MARK: - Private Properties

  @ObservationIgnored private var kokoroTTS: KokoroTTS?
  @ObservationIgnored private let playback = TTSPlaybackController(sampleRate: TTSProvider.kokoro.sampleRate)

  // MARK: - Initialization

  public init() {
    Log.tts.debug("KokoroEngine initialized")
  }

  // MARK: - TTSEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.tts.debug("KokoroEngine already loaded")
      return
    }

    Log.model.info("Loading Kokoro TTS model...")

    do {
      kokoroTTS = try await KokoroTTS.load(
        repoId: KokoroWeightLoader.defaultRepoId,
        progressHandler: progressHandler ?? { _ in },
      )

      isLoaded = true
      Log.model.info("Kokoro TTS model loaded successfully")
    } catch {
      Log.model.error("Failed to load Kokoro model: \(error.localizedDescription)")
      throw TTSError.modelLoadFailed(underlying: error)
    }
  }

  public func stop() async {
    await playback.stop(
      setGenerating: { self.isGenerating = $0 },
      setPlaying: { self.isPlaying = $0 },
    )
    Log.tts.debug("KokoroEngine stopped")
  }

  public func unload() async {
    await stop()
    kokoroTTS = nil
    isLoaded = false
    Log.tts.debug("KokoroEngine unloaded")
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
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: The generated audio result
  public func generate(
    _ text: String,
    voice: Voice,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.collectStream(
      generateStreaming(text, voice: voice, speed: speed),
    )

    Log.tts.timing("Kokoro generation", duration: processingTime)
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
  ///   - speed: Playback speed multiplier (default: 1.0)
  public func say(
    _ text: String,
    voice: Voice,
    speed: Float = 1.0,
  ) async throws {
    let audio = try await generate(text, voice: voice, speed: speed)
    await play(audio)
  }

  // MARK: - Streaming

  /// Generate audio as a stream of chunks (no playback)
  /// - Parameters:
  ///   - text: The text to synthesize
  ///   - voice: The voice to use
  ///   - speed: Playback speed multiplier (default: 1.0)
  /// - Returns: An async stream of audio chunks
  public func generateStreaming(
    _ text: String,
    voice: Voice,
    speed: Float = 1.0,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)

    guard !trimmedText.isEmpty else {
      return AsyncThrowingStream { $0.finish(throwing: TTSError.invalidArgument("Text cannot be empty")) }
    }

    let sampleRate = provider.sampleRate

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

      guard let kokoroTTS else {
        throw TTSError.modelNotLoaded
      }

      // Wrap model stream to convert [Float] -> AudioChunk
      let modelStream = try await kokoroTTS.generateStreaming(
        text: trimmedText,
        voice: voice,
        speed: speed,
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
  ///   - speed: Playback speed multiplier (default: 1.0)
  @discardableResult
  public func sayStreaming(
    _ text: String,
    voice: Voice,
    speed: Float = 1.0,
  ) async throws -> AudioResult {
    let (samples, processingTime) = try await playback.playStream(
      generateStreaming(text, voice: voice, speed: speed),
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

// MARK: - TTS Factory Extension

public extension TTS {
  /// Kokoro: 50+ voices with speed control
  static func kokoro() -> KokoroEngine {
    KokoroEngine()
  }
}
