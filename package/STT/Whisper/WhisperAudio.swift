// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

import Foundation
import MLX

// MARK: - Whisper Audio Constants

/// Whisper audio hyperparameters (hard-coded in the model)
///
/// Note: n_mels varies by model (80 for most models, 128 for large-v3-turbo)
/// and is loaded dynamically from config.json, not hardcoded here.
public enum WhisperAudio {
  public static let sampleRate = 16000
  public static let nFft = 400
  public static let hopLength = 160
  public static let chunkLength = 30 // seconds
  public static let nSamples = chunkLength * sampleRate // 480,000 samples in a 30-second chunk
  public static let nFrames = nSamples / hopLength // 3000 frames in a mel spectrogram input

  public static let nSamplesPerToken = hopLength * 2 // initial convolutions have stride 2
  public static let framesPerSecond = sampleRate / hopLength // 10ms per audio frame
  public static let tokensPerSecond = sampleRate / nSamplesPerToken // 20ms per audio token
}

// MARK: - Whisper-Specific Window Function

/// Create Hann window matching Python's implementation
/// Python: [0.5 * (1 - cos(2*pi*n/(size-1))) for n in range(size)]
private func whisperHannWindow(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray([1.0])
  }

  // Create array [0, 1, 2, ..., length-1]
  let indices = (0 ..< length).map { Float($0) }
  let n = MLXArray(indices)

  // Apply Hann window formula: 0.5 * (1 - cos(2*pi*n/(length-1)))
  let factor = 2.0 * Float.pi / Float(length - 1)
  return 0.5 * (1.0 - MLX.cos(n * factor))
}

// MARK: - Audio Preprocessing

/// Pad or trim the audio array to N_SAMPLES (30 seconds), as expected by the encoder.
///
/// - Parameters:
///   - array: Audio array to pad or trim
///   - length: Target length (default: 480,000 samples = 30 seconds at 16kHz)
/// - Returns: Audio array with exactly `length` samples
func padOrTrim(_ array: MLXArray, length: Int = WhisperAudio.nSamples) -> MLXArray {
  let n = array.shape[0]

  if n > length {
    // Trim to length
    return array[0 ..< length]
  } else if n < length {
    // Pad with zeros
    let padding = MLXArray.zeros([length - n]).asType(array.dtype)
    return MLX.concatenated([array, padding])
  } else {
    return array
  }
}

/// Compute the log-Mel spectrogram for Whisper.
///
/// Matches the Python implementation in mlx_audio/stt/models/whisper/audio.py
///
/// - Parameters:
///   - audio: Audio waveform (T,) in 16 kHz
///   - nMels: Number of Mel-frequency filters (varies by model: 80 for most, 128 for large-v3-turbo)
///   - padding: Number of zero samples to pad to the right
/// - Returns: Log-Mel spectrogram (n_frames, n_mels) - matches Python output shape
func whisperLogMelSpectrogram(
  audio: MLXArray,
  nMels: Int,
  padding: Int = 0
) -> MLXArray {
  var audioArray = audio

  if padding > 0 {
    audioArray = MLX.padded(audioArray, widths: [IntOrPair((0, padding))])
  }

  // Create Hann window (symmetric, matches Python's hanning function)
  // Python: [0.5 * (1 - cos(2*pi*n/(size-1))) for n in range(size)]
  let window = whisperHannWindow(length: WhisperAudio.nFft)

  // Compute STFT
  // stftResult shape: (T, F) = (n_frames, n_fft//2 + 1) = (n_frames, 201)
  let stftResult = stft(
    audioArray,
    window: window,
    nFft: WhisperAudio.nFft,
    hopLength: WhisperAudio.hopLength,
    winLength: WhisperAudio.nFft
  )

  // Python's audio.py does: freqs[:-1, :] which removes the last TIME FRAME (not frequency bin!)
  // stftResult is (n_frames, 201), so [:-1, :] gives (n_frames-1, 201)
  let freqs = stftResult[0 ..< (stftResult.shape[0] - 1), 0...]

  // Compute power spectrum (magnitude squared)
  // Python: magnitudes = freqs[:-1, :].abs().square()
  let magnitudes = MLX.pow(MLX.abs(freqs), 2) // (n_frames-1, 201)

  // Create mel filterbank using Slaney scale (matches Python Whisper)
  // Python uses mel_scale=None which defaults to slaney, with norm="slaney"
  // melFilters returns (n_mels, n_fft/2 + 1) = (n_mels, 201)
  let filters = melFilters(
    sampleRate: WhisperAudio.sampleRate,
    nFft: WhisperAudio.nFft,
    nMels: nMels,
    fMin: 0.0,
    fMax: 8000.0
  )

  // Apply mel filterbank: (T, F) @ (F, M) -> (T, M)
  // Python: mel_spec = magnitudes @ filters.T
  // magnitudes is (n_frames-1, 201)
  // filters.T is (201, n_mels)
  // result is (n_frames-1, n_mels)
  let melSpec = MLX.matmul(magnitudes, filters.transposed())

  // Log scale with clipping (Whisper uses max - 8.0 clipping)
  var logSpec = MLX.log10(MLX.maximum(melSpec, MLXArray(1e-10)))
  logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)

  // Normalize to roughly [-1, 1] range
  logSpec = (logSpec + 4.0) / 4.0

  return logSpec
}
