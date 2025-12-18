// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale

@preconcurrency import AVFoundation
import Foundation
import MLX
import Synchronization

/// Returns (sampleRate, audio).
func loadAudioArray(from url: URL) throws -> (Double, MLXArray) {
  let file = try AVAudioFile(forReading: url)

  let inFormat = file.processingFormat
  let totalFrames = AVAudioFrameCount(file.length)
  guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: totalFrames) else {
    throw AudioLoaderError.bufferAllocFailed
  }
  try file.read(into: inBuffer)

  if inFormat.commonFormat == .pcmFormatFloat32, let chans = inBuffer.floatChannelData {
    let frames = Int(inBuffer.frameLength)
    let channels: [[Float]] = (0 ..< Int(inFormat.channelCount)).map { c in
      let ptr = chans[c]
      return Array(UnsafeBufferPointer(start: ptr, count: frames))
    }
    return (inFormat.sampleRate, MLXArray(channels[0]))
  }

  let floatFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: inFormat.sampleRate,
                                  channels: inFormat.channelCount,
                                  interleaved: false)
  guard let floatFormat else {
    throw AudioLoaderError.floatFormatCreationFailed
  }
  guard let converter = AVAudioConverter(from: inFormat, to: floatFormat) else {
    throw AudioLoaderError.converterCreationFailed
  }
  guard let outBuffer = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: totalFrames) else {
    throw AudioLoaderError.outputBufferAllocFailed
  }

  let consumed = Atomic<Bool>(false)
  var convError: NSError?
  let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
    if consumed.exchange(true, ordering: .relaxed) {
      outStatus.pointee = .endOfStream
      return nil
    } else {
      outStatus.pointee = .haveData
      return inBuffer
    }
  }
  converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
  if let e = convError { throw e }

  let frames = Int(outBuffer.frameLength)
  guard let floatChannelData = outBuffer.floatChannelData else {
    throw AudioLoaderError.channelDataAccessFailed
  }
  let channels: [[Float]] = (0 ..< Int(floatFormat.channelCount)).map { c in
    let ptr = floatChannelData[c]
    return Array(UnsafeBufferPointer(start: ptr, count: frames))
  }
  return (floatFormat.sampleRate, MLXArray(channels[0]))
}

enum AudioLoaderError: LocalizedError {
  case bufferAllocFailed
  case outputBufferAllocFailed
  case floatFormatCreationFailed
  case converterCreationFailed
  case channelDataAccessFailed

  var errorDescription: String? {
    switch self {
      case .bufferAllocFailed:
        "Failed to allocate audio buffer"
      case .outputBufferAllocFailed:
        "Failed to allocate output buffer"
      case .floatFormatCreationFailed:
        "Failed to create float audio format"
      case .converterCreationFailed:
        "Failed to create audio converter"
      case .channelDataAccessFailed:
        "Failed to access float channel data"
    }
  }
}

enum WAVWriterError: LocalizedError {
  case noFrames
  case bufferAllocFailed
  case formatCreationFailed

  var errorDescription: String? {
    switch self {
      case .noFrames:
        "Audio has no frames to write"
      case .bufferAllocFailed:
        "Failed to allocate audio buffer"
      case .formatCreationFailed:
        "Failed to create audio format"
    }
  }
}

func saveAudioArray(_ audio: MLXArray, sampleRate: Double, to url: URL) throws {
  let frames = audio.shape[0]
  guard frames > 0 else { throw WAVWriterError.noFrames }

  guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(1), interleaved: false) else {
    throw WAVWriterError.formatCreationFailed
  }
  guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
    throw WAVWriterError.bufferAllocFailed
  }
  buffer.frameLength = AVAudioFrameCount(frames)

  let channels = [audio.asArray(Float32.self)]
  guard let dst = buffer.floatChannelData else {
    throw WAVWriterError.bufferAllocFailed
  }
  for (c, channel) in channels.enumerated() {
    channel.withUnsafeBufferPointer { src in
      guard let baseAddress = src.baseAddress else { return }
      dst[c].update(from: baseAddress, count: frames)
    }
  }

  let file = try AVAudioFile(forWriting: url, settings: format.settings)
  try file.write(from: buffer)
}
