// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLX

// MARK: - Fun-ASR Audio Constants

/// Fun-ASR audio hyperparameters
public enum FunASRAudio {
  public static let sampleRate = 16000
  public static let nFft = 400 // 25ms window at 16kHz
  public static let hopLength = 160 // 10ms hop
  public static let nMels = 80

  // LFR (Low Frame Rate) parameters
  public static let lfrM = 7 // Stack every 7 frames
  public static let lfrN = 6 // Subsample by factor of 6

  // Derived values
  public static let inputDim = nMels * lfrM // 560
  public static let framesPerSecond = sampleRate / hopLength // 100
}

// MARK: - Window Functions

/// Create a Hamming window (Fun-ASR uses Hamming, not Hann)
///
/// Formula: w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
///
/// - Parameter length: Window length
/// - Returns: Hamming window array
func hammingWindow(length: Int) -> MLXArray {
  if length == 1 {
    return MLXArray([1.0])
  }

  // Use MLX array creation directly (avoid Swift array intermediate)
  let n = MLXArray(0 ..< length).asType(.float32)
  let factor = 2.0 * Float.pi / Float(length - 1)

  return 0.54 - 0.46 * MLX.cos(n * factor)
}

// MARK: - Mel Spectrogram

/// Compute log-mel spectrogram for Fun-ASR
///
/// - Parameters:
///   - audio: Audio waveform (T,) at 16kHz
///   - nMels: Number of mel filterbank bins (default: 80)
///   - nFft: FFT size (default: 400)
///   - hopLength: Hop length (default: 160)
/// - Returns: Log-mel spectrogram (n_frames, n_mels)
func funASRLogMelSpectrogram(
  audio: MLXArray,
  nMels: Int = FunASRAudio.nMels,
  nFft: Int = FunASRAudio.nFft,
  hopLength: Int = FunASRAudio.hopLength
) -> MLXArray {
  // Create Hamming window (Fun-ASR uses Hamming)
  let window = hammingWindow(length: nFft)

  // Compute STFT
  let stftResult = funASRSTFT(
    audio,
    window: window,
    nFft: nFft,
    hopLength: hopLength
  )

  // Compute power spectrum (magnitude squared)
  // Remove last frequency bin to get (n_frames, n_fft/2)
  let freqs = stftResult[0..., 0 ..< (nFft / 2)]
  let magnitudes = MLX.pow(MLX.abs(freqs), 2)

  // Create mel filterbank with HTK scale (Fun-ASR uses htk)
  let filters = funASRMelFilters(
    sampleRate: FunASRAudio.sampleRate,
    nFft: nFft,
    nMels: nMels,
    melScale: .htk
  )

  // Apply mel filterbank: (T, F) @ (F, M) -> (T, M)
  let melSpec = MLX.matmul(magnitudes, filters.transposed())

  // Log compression (natural log, matching Python)
  let logSpec = MLX.log(MLX.maximum(melSpec, MLXArray(1e-10)))

  return logSpec
}

// MARK: - LFR Processing

/// Apply Low Frame Rate (LFR) processing to features
///
/// This stacks consecutive frames and subsamples to reduce the frame rate.
/// Uses vectorized gather operations for efficiency.
///
/// - Parameters:
///   - features: Input mel spectrogram (n_frames, n_mels)
///   - lfrM: Number of frames to stack (default: 7)
///   - lfrN: Subsampling factor (default: 6)
/// - Returns: LFR-processed features (ceil(n_frames / lfrN), n_mels * lfrM)
func applyLFR(
  _ features: MLXArray,
  lfrM: Int = FunASRAudio.lfrM,
  lfrN: Int = FunASRAudio.lfrN
) -> MLXArray {
  let T = features.shape[0]
  let nMels = features.shape[1]

  // Output length uses ceiling division
  let tLFR = Int(ceil(Double(T) / Double(lfrN)))

  // Left padding with first frame repeated
  let leftPad = (lfrM - 1) / 2
  var paddedFeatures = features

  if leftPad > 0 {
    // Broadcast first frame to create left padding
    let firstFrame = features[0].expandedDimensions(axis: 0)
    let leftPadding = MLX.broadcast(firstFrame, to: [leftPad, nMels])
    paddedFeatures = MLX.concatenated([leftPadding, paddedFeatures], axis: 0)
  }

  // Right padding to ensure we have enough frames
  let tPadded = paddedFeatures.shape[0]
  let totalNeeded = (tLFR - 1) * lfrN + lfrM
  if totalNeeded > tPadded {
    let rightPad = totalNeeded - tPadded
    let lastFrame = paddedFeatures[tPadded - 1].expandedDimensions(axis: 0)
    let rightPadding = MLX.broadcast(lastFrame, to: [rightPad, nMels])
    paddedFeatures = MLX.concatenated([paddedFeatures, rightPadding], axis: 0)
  }

  // Create indices for all output frames using vectorized gather
  // startIndices: [0, lfrN, 2*lfrN, ..., (tLFR-1)*lfrN]
  let startIndices = MLXArray(0 ..< tLFR) * lfrN
  // offsets: [0, 1, 2, ..., lfrM-1]
  let offsets = MLXArray(0 ..< lfrM)

  // Broadcasting: (tLFR, 1) + (lfrM,) -> (tLFR, lfrM)
  let indices = startIndices.expandedDimensions(axis: 1) + offsets.expandedDimensions(axis: 0)

  // Gather frames: paddedFeatures[indices] -> (tLFR, lfrM, nMels)
  let gathered = paddedFeatures[indices]

  // Reshape to (tLFR, lfrM * nMels)
  return gathered.reshaped([tLFR, lfrM * nMels])
}

// MARK: - CMVN Normalization

/// Apply Cepstral Mean and Variance Normalization (CMVN)
///
/// - Parameters:
///   - features: Input features (T, D)
///   - cmvnMean: Precomputed mean shift (optional)
///   - cmvnIstd: Precomputed inverse std (optional)
/// - Returns: Normalized features
func applyCMVN(
  _ features: MLXArray,
  cmvnMean: MLXArray? = nil,
  cmvnIstd: MLXArray? = nil
) -> MLXArray {
  if let mean = cmvnMean, let istd = cmvnIstd {
    // Apply precomputed CMVN: (x + mean) * istd
    // Note: cmvnMean is actually the negative mean (shift)
    return (features + mean) * istd
  }

  // Per-utterance normalization
  let mean = features.mean(axis: 0, keepDims: true)
  let std = MLX.variance(features, axis: 0, keepDims: true).sqrt() + 1e-6
  return (features - mean) / std
}

// MARK: - Full Preprocessing Pipeline

/// Full audio preprocessing pipeline for Fun-ASR
///
/// 1. Compute log mel spectrogram
/// 2. Apply LFR (frame stacking and subsampling)
/// 3. Apply CMVN normalization
///
/// - Parameters:
///   - audio: Input audio waveform (T,) at 16kHz
///   - nMels: Number of mel bins (default: 80)
///   - lfrM: LFR frame stacking count (default: 7)
///   - lfrN: LFR subsampling factor (default: 6)
///   - applyNormalization: Whether to apply CMVN (default: true)
/// - Returns: Preprocessed features (ceil(T / (hopLength * lfrN)), nMels * lfrM)
func preprocessAudio(
  _ audio: MLXArray,
  nMels: Int = FunASRAudio.nMels,
  lfrM: Int = FunASRAudio.lfrM,
  lfrN: Int = FunASRAudio.lfrN,
  applyNormalization: Bool = true
) -> MLXArray {
  // Compute log mel spectrogram
  var features = funASRLogMelSpectrogram(audio: audio, nMels: nMels)

  // Apply LFR processing
  features = applyLFR(features, lfrM: lfrM, lfrN: lfrN)

  // Apply normalization
  if applyNormalization {
    features = applyCMVN(features)
  }

  return features
}

/// Compute output feature lengths after preprocessing
///
/// - Parameters:
///   - audioLength: Length of input audio in samples
///   - hopLength: Hop length for STFT (default: 160)
///   - lfrN: LFR subsampling factor (default: 6)
/// - Returns: Output feature length
func computeFeatureLength(
  audioLength: Int,
  hopLength: Int = FunASRAudio.hopLength,
  lfrN: Int = FunASRAudio.lfrN
) -> Int {
  // Frames after STFT (approximately)
  let nFrames = audioLength / hopLength

  // Frames after LFR (ceiling division)
  return (nFrames + lfrN - 1) / lfrN
}

// MARK: - Private Helper Functions

/// Compute STFT of a signal
private func funASRSTFT(
  _ x: MLXArray,
  window: MLXArray,
  nFft: Int,
  hopLength: Int,
  center: Bool = true
) -> MLXArray {
  var xArray = x

  // Pad window to nFft if needed
  var w = window
  if w.shape[0] < nFft {
    let padSize = nFft - w.shape[0]
    w = MLX.concatenated([w, MLXArray.zeros([padSize])])
  }

  // Center padding with reflection
  if center {
    xArray = reflectPad1D(xArray, padding: nFft / 2)
  }

  // Calculate number of frames
  let numFrames = 1 + (xArray.shape[0] - nFft) / hopLength
  guard numFrames > 0 else {
    fatalError("Input is too short for STFT")
  }

  // Create frames using as_strided
  let shape = [numFrames, nFft]
  let strides = [hopLength, 1]
  let frames = MLX.asStrided(xArray, shape, strides: strides)

  // Apply window and compute FFT
  let windowedFrames = frames * w
  let spec = MLX.rfft(windowedFrames)

  return spec
}

/// Reflect padding for 1D array
private func reflectPad1D(_ x: MLXArray, padding: Int) -> MLXArray {
  if padding == 0 {
    return x
  }

  let n = x.shape[0]
  if n == 1 {
    return MLX.concatenated([
      MLXArray.full([padding], values: x[0]),
      x,
      MLXArray.full([padding], values: x[0]),
    ])
  }

  // Reflect at boundaries
  var prefixArray = reverseAlongAxis(x[1 ..< min(padding + 1, n)], axis: 0)
  var suffixArray = reverseAlongAxis(x[max(0, n - padding - 1) ..< (n - 1)], axis: 0)

  // Handle cases where array is shorter than padding
  while prefixArray.shape[0] < padding {
    let additional = min(padding - prefixArray.shape[0], n - 1)
    prefixArray = MLX.concatenated([reverseAlongAxis(x[1 ..< (additional + 1)], axis: 0), prefixArray])
  }

  while suffixArray.shape[0] < padding {
    let additional = min(padding - suffixArray.shape[0], n - 1)
    suffixArray = MLX.concatenated([suffixArray, reverseAlongAxis(x[(n - additional - 1) ..< (n - 1)], axis: 0)])
  }

  return MLX.concatenated([prefixArray[0 ..< padding], x, suffixArray[0 ..< padding]])
}

/// Mel scale type
private enum MelScale {
  case slaney
  case htk
}

/// Create mel filterbank using vectorized MLX operations
///
/// This is optimized to use MLX operations for parallel computation on GPU,
/// avoiding CPU loops for better performance.
private func funASRMelFilters(
  sampleRate: Int,
  nFft: Int,
  nMels: Int,
  fMin: Float = 0.0,
  fMax: Float? = nil,
  melScale: MelScale = .htk
) -> MLXArray {
  let actualFMax = fMax ?? Float(sampleRate) / 2.0

  // Vectorized mel scale conversion functions
  func hzToMel(_ hz: Float) -> Float {
    switch melScale {
      case .htk:
        return 2595.0 * log10(1.0 + hz / 700.0)
      case .slaney:
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep: Float = log(6.4) / 27.0
        return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : hz / fSp
    }
  }

  func melToHzVectorized(_ mels: MLXArray) -> MLXArray {
    switch melScale {
      case .htk:
        // HTK formula: 700 * (10^(mel / 2595) - 1)
        return 700.0 * (MLX.pow(MLXArray(10.0), mels / 2595.0) - 1.0)
      case .slaney:
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep: Float = log(6.4) / 27.0
        let linear = fSp * mels
        let logarithmic = minLogHz * MLX.exp(logstep * (mels - minLogMel))
        return MLX.where(mels .>= minLogMel, logarithmic, linear)
    }
  }

  // Generate frequency points using MLX linspace
  let nFreqs = nFft / 2
  let allFreqs = MLX.linspace(Float(0), Float(sampleRate) / 2.0, count: nFreqs)

  // Convert frequencies to mel and back to hz using vectorized operations
  let mMin = hzToMel(fMin)
  let mMax = hzToMel(actualFMax)
  let mPts = MLX.linspace(mMin, mMax, count: nMels + 2)
  let fPts = melToHzVectorized(mPts)

  // Compute slopes for filterbank using broadcasting
  // Python: slopes = f_pts[None, :] - all_freqs[:, None]
  // slopes[i, j] = f_pts[j] - all_freqs[i]
  let fDiff = fPts[1...] - fPts[..<(-1)]
  let slopes = fPts.expandedDimensions(axis: 0) - allFreqs.expandedDimensions(axis: 1)

  // Calculate overlapping triangular filters
  // down_slopes = -slopes[:, :-2] / f_diff[:-1]  (rising edge)
  // up_slopes = slopes[:, 2:] / f_diff[1:]       (falling edge)
  let downSlopes = -slopes[0..., ..<(-2)] / fDiff[..<(-1)]
  let upSlopes = slopes[0..., 2...] / fDiff[1...]

  // filterbank = max(0, min(down_slopes, up_slopes))
  var filterbank = MLX.maximum(
    MLXArray.zeros(like: downSlopes),
    MLX.minimum(downSlopes, upSlopes)
  )

  // Apply Slaney normalization
  let enorm = 2.0 / (fPts[2 ..< (nMels + 2)] - fPts[0 ..< nMels])
  filterbank = filterbank * enorm.expandedDimensions(axis: 0)

  // Transpose: (nFreqs, nMels) -> (nMels, nFreqs)
  return filterbank.transposed()
}
