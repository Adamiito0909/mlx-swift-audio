// Copyright © Anthony DePasquale

@preconcurrency import AVFoundation
import MLX
import Synchronization

/// High-quality audio resampling using AVAudioConverter with anti-aliasing.
public enum AudioResampler {
  /// Resample audio from one sample rate to another.
  /// - Parameters:
  ///   - audio: Input audio samples as MLXArray
  ///   - sourceSampleRate: Original sample rate in Hz
  ///   - targetSampleRate: Target sample rate in Hz
  /// - Returns: Resampled audio as MLXArray
  public static func resample(
    _ audio: MLXArray,
    from sourceSampleRate: Int,
    to targetSampleRate: Int,
  ) -> MLXArray {
    if sourceSampleRate == targetSampleRate {
      return audio
    }

    let inputSamples = audio.asArray(Float.self)

    guard let inputFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: Double(sourceSampleRate),
      channels: 1,
      interleaved: false,
    ),
      let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(targetSampleRate),
        channels: 1,
        interleaved: false,
      )
    else {
      fatalError("Failed to create audio format for resampling \(sourceSampleRate) -> \(targetSampleRate)")
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
      fatalError("Failed to create AVAudioConverter for resampling \(sourceSampleRate) -> \(targetSampleRate)")
    }

    let frameCount = AVAudioFrameCount(inputSamples.count)
    guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
      fatalError("Failed to create input buffer for resampling")
    }
    inputBuffer.frameLength = frameCount

    if let channelData = inputBuffer.floatChannelData {
      inputSamples.withUnsafeBufferPointer { ptr in
        channelData[0].initialize(from: ptr.baseAddress!, count: inputSamples.count)
      }
    }

    let outputFrameCount = AVAudioFrameCount(
      Double(inputSamples.count) * Double(targetSampleRate) / Double(sourceSampleRate) + 1)
    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount)
    else {
      fatalError("Failed to create output buffer for resampling")
    }

    var error: NSError?
    let inputConsumed = Atomic<Bool>(false)

    let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
      if inputConsumed.exchange(true, ordering: .relaxed) {
        outStatus.pointee = .noDataNow
        return nil
      }
      outStatus.pointee = .haveData
      return inputBuffer
    }

    if status == .error || error != nil {
      fatalError("Audio resampling failed: \(error?.localizedDescription ?? "unknown error")")
    }

    guard let outputChannelData = outputBuffer.floatChannelData else {
      fatalError("Failed to get output channel data after resampling")
    }

    let outputLength = Int(outputBuffer.frameLength)
    let resampled = Array(UnsafeBufferPointer(start: outputChannelData[0], count: outputLength))
    return MLXArray(resampled)
  }
}
