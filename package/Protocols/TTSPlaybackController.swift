// Copyright © Anthony DePasquale

import Foundation

/// Manages audio playback and generation task lifecycle for TTS engines.
///
/// Compose this controller into TTS engines to eliminate duplicated code for:
/// - Audio playback with state management
/// - Generation task cancellation and tracking
/// - Streaming playback coordination
/// - Collecting stream results
///
/// Example usage:
/// ```swift
/// @Observable
/// public final class MyEngine: TTSEngine {
///   private let playback: TTSPlaybackController
///
///   public func stop() async {
///     await playback.stop(
///       setGenerating: { isGenerating = $0 },
///       setPlaying: { isPlaying = $0 }
///     )
///   }
/// }
/// ```
@MainActor
public final class TTSPlaybackController {
  // MARK: - Private Properties

  private let audioPlayer: AudioSamplePlayer
  private var generationTask: Task<Void, Never>?

  // MARK: - Initialization

  public init(sampleRate: Int) {
    audioPlayer = AudioSamplePlayer(sampleRate: sampleRate)
  }

  // MARK: - Stop Operations

  /// Stop generation and playback, updating state via closures
  public func stop(
    setGenerating: (Bool) -> Void,
    setPlaying: (Bool) -> Void,
  ) async {
    generationTask?.cancel()
    generationTask = nil
    setGenerating(false)
    await audioPlayer.stop()
    setPlaying(false)
  }

  // MARK: - Playback Operations

  /// Play audio samples with state management
  public func play(
    _ audio: AudioResult,
    setPlaying: (Bool) -> Void,
  ) async {
    guard case let .samples(samples, _, _) = audio else {
      Log.audio.warning("Cannot play AudioResult.file - use AudioFilePlayer instead")
      return
    }
    setPlaying(true)
    await audioPlayer.play(samples: samples)
    setPlaying(false)
  }

  /// Enqueue samples for streaming playback
  public func enqueue(samples: [Float], prebufferSeconds: Double = 0) {
    audioPlayer.enqueue(samples: samples, prebufferSeconds: prebufferSeconds)
  }

  /// Stop audio playback only (without affecting generation task)
  public func stopAudio() async {
    await audioPlayer.stop()
  }

  /// Wait for all enqueued audio to finish playing
  public func awaitAudioDrain() async {
    await audioPlayer.awaitDrain()
  }

  // MARK: - Audio File Output

  /// Save audio samples to file
  ///
  /// - Parameters:
  ///   - samples: Audio samples to save
  ///   - sampleRate: Sample rate of the audio
  /// - Returns: URL of the saved file, or nil if save failed
  public func saveAudioFile(samples: [Float], sampleRate: Int) -> URL? {
    guard !samples.isEmpty else { return nil }
    do {
      return try AudioFileWriter.save(
        samples: samples,
        sampleRate: sampleRate,
        filename: TTSConstants.outputFilename,
      )
    } catch {
      Log.audio.error("Failed to save audio file: \(error.localizedDescription)")
      return nil
    }
  }

  // MARK: - Stream Collection

  /// Collect all chunks from a generation stream into samples
  ///
  /// Use this to implement non-streaming `generate()` methods.
  public func collectStream(
    _ stream: AsyncThrowingStream<AudioChunk, Error>,
  ) async throws -> (samples: [Float], processingTime: TimeInterval) {
    var allSamples: [Float] = []
    var processingTime: TimeInterval = 0

    for try await chunk in stream {
      allSamples.append(contentsOf: chunk.samples)
      processingTime = chunk.processingTime
    }

    return (allSamples, processingTime)
  }

  /// Play a generation stream as chunks arrive (streaming playback)
  ///
  /// Use this to implement `sayStreaming()` methods.
  /// Note: `isGenerating` state is managed by `createGenerationStream()`, not here.
  public func playStream(
    _ stream: AsyncThrowingStream<AudioChunk, Error>,
    setPlaying: (Bool) -> Void,
  ) async throws -> (samples: [Float], processingTime: TimeInterval) {
    await stopAudio()
    setPlaying(true)

    var allSamples: [Float] = []
    var totalProcessingTime: TimeInterval = 0

    do {
      for try await chunk in stream {
        allSamples.append(contentsOf: chunk.samples)
        totalProcessingTime = chunk.processingTime
        enqueue(samples: chunk.samples, prebufferSeconds: 0)
      }

      // Wait for all enqueued audio to finish playing
      await awaitAudioDrain()
      setPlaying(false)

      return (allSamples, totalProcessingTime)
    } catch {
      setPlaying(false)
      await stopAudio()
      throw error
    }
  }

  // MARK: - Generation Stream Wrapper

  /// Create a managed generation stream with proper task lifecycle
  ///
  /// This wraps a model's streaming output with:
  /// - Automatic cancellation of previous generation
  /// - Task tracking for stop() support
  /// - Proper cleanup on termination
  /// - CancellationError handling
  ///
  /// - Parameters:
  ///   - setGenerating: Closure to update isGenerating state
  ///   - setGenerationTime: Closure to update generationTime (called on first chunk)
  ///   - generator: Async closure that yields audio chunks
  public func createGenerationStream(
    setGenerating: @escaping (Bool) -> Void,
    setGenerationTime: @escaping (TimeInterval) -> Void,
    generator: @escaping () async throws -> AsyncThrowingStream<AudioChunk, Error>,
  ) -> AsyncThrowingStream<AudioChunk, Error> {
    AsyncThrowingStream { continuation in
      // Cancel any existing generation
      self.generationTask?.cancel()

      self.generationTask = Task { @MainActor [weak self] in
        guard self != nil else {
          continuation.finish()
          return
        }

        setGenerating(true)
        let startTime = Date()
        var isFirst = true

        do {
          let stream = try await generator()

          for try await chunk in stream {
            guard !Task.isCancelled else { break }

            if isFirst {
              setGenerationTime(Date().timeIntervalSince(startTime))
              isFirst = false
            }

            continuation.yield(chunk)
          }

          setGenerating(false)
          self?.generationTask = nil
          continuation.finish()

        } catch is CancellationError {
          setGenerating(false)
          self?.generationTask = nil
          continuation.finish(throwing: CancellationError())
        } catch {
          setGenerating(false)
          self?.generationTask = nil
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { [weak self] _ in
        Task { @MainActor in
          self?.generationTask?.cancel()
        }
      }
    }
  }
}
