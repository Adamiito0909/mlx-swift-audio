import Foundation
import MLX
import Synchronization

/// Actor wrapper for Whisper model that provides thread-safe transcription
actor WhisperSTT {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  nonisolated(unsafe) let model: WhisperModel
  nonisolated(unsafe) let tokenizer: WhisperTokenizer

  // MARK: - Initialization

  private init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  /// Load WhisperSTT from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - modelSize: Model size to load
  ///   - quantization: Quantization level (fp16, 8bit, 4bit). Default is 4bit.
  ///   - progressHandler: Optional callback for download/load progress
  /// - Returns: Initialized WhisperSTT instance
  static func load(
    modelSize: WhisperModelSize,
    quantization: WhisperQuantization = .q4,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> WhisperSTT {
    // Load model first (the slow operation with progress)
    let model = try await WhisperModel.load(
      modelSize: modelSize,
      quantization: quantization,
      progressHandler: progressHandler
    )

    // Then load tokenizer (fast operation) with correct vocabulary for model type
    // Pass model directory so tokenizer can find vocab files bundled with the model
    let tokenizer = try await WhisperTokenizer.load(
      isMultilingual: model.isMultilingual,
      modelDirectory: model.modelDirectory
    )

    // Validate tokenizer configuration matches model expectations
    // This catches critical bugs like off-by-one errors in special token IDs
    let modelVocabSize = model.dims.n_vocab

    // Verify key special tokens are in valid range
    let maxTokenId = max(
      tokenizer.eot,
      tokenizer.sot,
      tokenizer.translate,
      tokenizer.transcribe,
      tokenizer.noSpeech,
      tokenizer.timestampBegin
    )

    if maxTokenId >= modelVocabSize {
      throw STTError.invalidArgument(
        """
        Tokenizer misconfiguration: token ID \(maxTokenId) >= model vocab size \(modelVocabSize). \
        This indicates a critical bug in tokenizer setup.
        """
      )
    }

    // Verify critical token IDs match expected values based on model type
    // Multilingual: eot=50257, sot=50258, transcribe=50359
    // English-only: eot=50256, sot=50257, transcribe=50358
    if model.isMultilingual {
      assert(tokenizer.eot == 50257, "Multilingual EOT token must be 50257")
      assert(tokenizer.sot == 50258, "Multilingual SOT token must be 50258")
      assert(tokenizer.transcribe == 50359, "Multilingual transcribe token must be 50359")
    } else {
      assert(tokenizer.eot == 50256, "English-only EOT token must be 50256")
      assert(tokenizer.sot == 50257, "English-only SOT token must be 50257")
      assert(tokenizer.transcribe == 50358, "English-only transcribe token must be 50358")
    }

    return WhisperSTT(model: model, tokenizer: tokenizer)
  }

  // MARK: - Transcription

  /// Transcribe audio to text using seek-based processing (matching Python implementation)
  ///
  /// This uses a seek pointer to move through the audio, with content-aware advancement
  /// based on decoded timestamps and word boundaries. This matches Python's implementation
  /// and provides better handling of long audio with silence or boundary cases.
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) in 16 kHz
  ///   - language: Optional language code (e.g., "en", "zh"), nil for auto-detect
  ///   - task: Transcription task (transcribe or translate)
  ///   - temperature: Sampling temperature (0.0 for greedy)
  ///   - timestamps: Timestamp granularity
  ///   - conditionOnPreviousText: Whether to use previous segment's output as prompt (default: true)
  ///   - noSpeechThreshold: Skip segments with no_speech_prob > threshold (default: 0.6)
  ///   - logprobThreshold: Skip if avg_logprob < threshold (default: -1.0)
  ///   - compressionRatioThreshold: Retry with higher temperature if compression ratio > threshold (default: 2.4)
  ///     High compression ratio indicates repetitive text (potential hallucination).
  ///   - hallucinationSilenceThreshold: When word timestamps are enabled, skip silent periods
  ///     longer than this threshold (in seconds) when a possible hallucination is detected.
  ///     Set to nil (default) to disable hallucination filtering.
  /// - Returns: Transcription result
  func transcribe(
    audio: MLXArray,
    language: String?,
    task: TranscriptionTask,
    temperature: Float,
    timestamps: TimestampGranularity,
    conditionOnPreviousText: Bool = true,
    noSpeechThreshold: Float? = 0.6,
    logprobThreshold: Float? = -1.0,
    compressionRatioThreshold: Float? = 2.4,
    hallucinationSilenceThreshold: Float? = nil
  ) -> TranscriptionResult {
    let transcribeStartTime = CFAbsoluteTimeGetCurrent()

    // Constants matching Python
    let nFrames = WhisperAudio.nFrames // 3000 frames per 30s segment
    let hopLength = WhisperAudio.hopLength // 160
    let sampleRate = WhisperAudio.sampleRate // 16000
    let framesPerSecond = WhisperAudio.framesPerSecond // 100
    let inputStride = nFrames / model.dims.n_audio_ctx // mel frames per output token: 2
    let timePrecision = Float(inputStride * hopLength) / Float(sampleRate) // 0.02 seconds per token

    // Pad audio with 30 seconds of silence for boundary handling
    let paddedAudio = MLX.concatenated([audio, MLXArray.zeros([WhisperAudio.nSamples])], axis: 0)

    // Compute mel spectrogram for entire audio (with padding)
    let fullMel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    // Transpose from (n_mels, n_frames) to (n_frames, n_mels) for MLX Conv1d
    let fullMelTransposed = fullMel.transposed()
    eval(fullMelTransposed)

    // Content frames (excluding padding)
    let contentFrames = audio.shape[0] / hopLength
    let contentDuration = Float(contentFrames * hopLength) / Float(sampleRate)

    Log.model.info("Transcribing \(String(format: "%.1f", contentDuration))s audio with seek-based processing")

    // Detect language if not specified
    var detectedLanguage: String? = nil
    if language == nil {
      let melSegment = padOrTrimMel(fullMelTransposed[0 ..< nFrames], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0)
      let (lang, prob) = detectLanguageFromMel(batchedMel)
      detectedLanguage = lang
      Log.model.info("Detected language: \(lang) (probability: \(String(format: "%.2f", prob)))")
    }
    let languageToUse = language ?? detectedLanguage ?? "en"

    // Seek-based transcription loop
    var seek = 0
    var allTokens: [Int] = []
    var allSegments: [TranscriptionSegment] = []
    var promptResetSince = 0
    var lastSpeechTimestamp: Float = 0.0

    while seek < contentFrames {
      let timeOffset = Float(seek * hopLength) / Float(sampleRate)
      let windowEndTime = Float((seek + nFrames) * hopLength) / Float(sampleRate)
      let segmentSize = min(nFrames, contentFrames - seek)
      let segmentDuration = Float(segmentSize * hopLength) / Float(sampleRate)

      // Extract mel segment and pad to nFrames
      let melSegment = padOrTrimMel(fullMelTransposed[seek ..< (seek + segmentSize)], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0)

      // Build prompt from previous tokens (if conditioning enabled)
      // Use tokens since last prompt reset (matches Python: all_tokens[prompt_reset_since:])
      let promptTokens = conditionOnPreviousText ? Array(allTokens[promptResetSince...]) : []
      let prompt = promptTokens

      // Temperature fallback loop (matches Python's decode_with_fallback)
      // Try increasing temperatures when output is too repetitive (high compression ratio)
      // or has low confidence (low avg_logprob)
      let temperatureFallbackSequence: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
      var result: DecodingResult!

      for currentTemperature in temperatureFallbackSequence {
        // Create decoding options with current temperature
        let options = DecodingOptions(
          task: task,
          language: languageToUse,
          temperature: currentTemperature,
          maxTokens: 448,
          timestamps: timestamps,
          prompt: prompt
        )

        // Decode segment
        let decoder = GreedyDecoder(model: model, tokenizer: tokenizer, options: options)
        result = decoder.decode(batchedMel)

        // Check if we need to retry with higher temperature
        var needsFallback = false

        // Too repetitive (high compression ratio indicates hallucination)
        if let crThreshold = compressionRatioThreshold,
           result.compressionRatio > crThreshold
        {
          needsFallback = true
          Log.model.debug("Compression ratio \(String(format: "%.2f", result.compressionRatio)) > \(crThreshold), retrying with higher temperature")
        }

        // Too low confidence
        if let lpThreshold = logprobThreshold,
           result.avgLogProb < lpThreshold
        {
          needsFallback = true
          Log.model.debug("Avg log prob \(String(format: "%.2f", result.avgLogProb)) < \(lpThreshold), retrying with higher temperature")
        }

        // If it's likely silence, accept the result and don't retry
        // Python: if no_speech_prob > no_speech_threshold: needs_fallback = False
        if let nsThreshold = noSpeechThreshold,
           result.noSpeechProb > nsThreshold
        {
          needsFallback = false
        }

        if !needsFallback {
          break
        }

        // If we're at the last temperature, use whatever we got
        if currentTemperature == temperatureFallbackSequence.last {
          Log.model.warning("All temperature fallbacks exhausted, using final result")
        }
      }

      // No-speech detection: skip if no_speech_prob > threshold
      if let nsThreshold = noSpeechThreshold {
        var shouldSkip = result.noSpeechProb > nsThreshold

        // Don't skip if logprob is high enough despite high no_speech_prob
        if let lpThreshold = logprobThreshold, result.avgLogProb > lpThreshold {
          shouldSkip = false
        }

        if shouldSkip {
          seek += segmentSize
          continue
        }
      }

      let previousSeek = seek
      var currentSegments: [TranscriptionSegment] = []

      // Parse tokens to extract segments based on timestamps
      let tokens = result.tokens
      let timestampTokens = tokens.map { $0 >= tokenizer.timestampBegin }

      // Find consecutive timestamp pairs
      var consecutiveIndices: [Int] = []
      if timestampTokens.count >= 2 {
        for i in 0 ..< (timestampTokens.count - 1) {
          if timestampTokens[i], timestampTokens[i + 1] {
            consecutiveIndices.append(i + 1)
          }
        }
      }

      // Check for single timestamp ending
      let singleTimestampEnding = timestampTokens.count >= 2 &&
        !timestampTokens[timestampTokens.count - 2] &&
        timestampTokens[timestampTokens.count - 1]

      if !consecutiveIndices.isEmpty {
        // Multiple segments based on consecutive timestamps
        var slices = consecutiveIndices
        if singleTimestampEnding {
          slices.append(tokens.count)
        }

        var lastSlice = 0
        for currentSlice in slices {
          let slicedTokens = Array(tokens[lastSlice ..< currentSlice])
          guard slicedTokens.count >= 2 else {
            lastSlice = currentSlice
            continue
          }

          let startTimestampPos = slicedTokens[0] - tokenizer.timestampBegin
          let endTimestampPos = slicedTokens[slicedTokens.count - 1] - tokenizer.timestampBegin

          let segmentStart = timeOffset + Float(startTimestampPos) * timePrecision
          let segmentEnd = timeOffset + Float(endTimestampPos) * timePrecision

          // Extract text tokens for this slice
          let textTokens = slicedTokens.filter { $0 < tokenizer.eot }
          let text = tokenizer.decode(textTokens)

          let segment = TranscriptionSegment(
            text: text,
            start: TimeInterval(segmentStart),
            end: TimeInterval(segmentEnd),
            tokens: slicedTokens,
            avgLogProb: result.avgLogProb,
            noSpeechProb: result.noSpeechProb,
            words: nil
          )
          currentSegments.append(segment)

          lastSlice = currentSlice
        }

        // Advance seek based on timestamps
        // When single_timestamp_ending and there's remaining audio,
        // advance to the timestamp position instead of full segment to avoid
        // skipping content in short audio clips.
        if singleTimestampEnding {
          if let lastTimestamp = tokens.last, lastTimestamp != tokenizer.timestampBegin {
            let lastTimestampPos = lastTimestamp - tokenizer.timestampBegin
            let timestampSeek = lastTimestampPos * inputStride
            if seek + timestampSeek < contentFrames {
              seek += timestampSeek
            } else {
              seek += segmentSize
            }
          } else {
            seek += segmentSize
          }
        } else {
          let lastTimestampPos = tokens[consecutiveIndices.last! - 1] - tokenizer.timestampBegin
          seek += lastTimestampPos * inputStride
        }
      } else {
        // Single segment (no consecutive timestamps)
        // Python: duration = segment_duration, then check for last timestamp
        var duration = segmentDuration

        // Find last timestamp token if any
        // Python: timestamps = tokens[timestamp_tokens.nonzero()[0]]
        let timestampIndices = tokens.enumerated().compactMap { i, t in t >= tokenizer.timestampBegin ? i : nil }
        if let lastIdx = timestampIndices.last, tokens[lastIdx] != tokenizer.timestampBegin {
          // Python: last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
          let lastTimestampPos = tokens[lastIdx] - tokenizer.timestampBegin
          duration = Float(lastTimestampPos) * timePrecision
        }

        let textTokens = tokens.filter { $0 < tokenizer.eot }
        let text = tokenizer.decode(textTokens)

        let segment = TranscriptionSegment(
          text: text,
          start: TimeInterval(timeOffset),
          end: TimeInterval(timeOffset + duration),
          tokens: tokens,
          avgLogProb: result.avgLogProb,
          noSpeechProb: result.noSpeechProb,
          words: nil
        )
        currentSegments.append(segment)

        // Python: seek += segment_size (ALWAYS advance by full segment, not duration)
        // The duration is only used for segment end time, not seek advancement
        //
        // When there's a single timestamp ending and remaining audio exists,
        // advance to the timestamp position instead of full segment to avoid
        // skipping content in short audio clips.
        if singleTimestampEnding, let lastIdx = timestampIndices.last, tokens[lastIdx] != tokenizer.timestampBegin {
          let lastTimestampPos = tokens[lastIdx] - tokenizer.timestampBegin
          let timestampSeek = lastTimestampPos * inputStride
          // Only use timestamp-based seek if there's remaining audio
          if seek + timestampSeek < contentFrames {
            seek += timestampSeek
          } else {
            seek += segmentSize
          }
        } else {
          seek += segmentSize
        }
      }

      // Add word timestamps if requested (batched for efficiency)
      if timestamps == .word {
        // Use batched word timestamp extraction (single forward pass for all segments)
        lastSpeechTimestamp = addWordTimestamps(
          segments: &currentSegments,
          model: model,
          tokenizer: tokenizer,
          mel: batchedMel,
          numFrames: segmentSize,
          language: languageToUse,
          task: task,
          timeOffset: timeOffset,
          lastSpeechTimestamp: lastSpeechTimestamp
        )

        // Content-aware seek advancement based on last word
        if !singleTimestampEnding {
          if let lastWordEnd = getLastWordEnd(currentSegments), lastWordEnd > timeOffset {
            seek = Int(lastWordEnd * Float(framesPerSecond))
          }
        }

        // Hallucination detection (inline, matching Python)
        if let threshold = hallucinationSilenceThreshold {
          // Python lines 756-767: Check remaining duration after last word
          // If remaining silence > threshold, keep the last_word_end seek
          // Otherwise, reset to previous_seek + segment_size
          if !singleTimestampEnding {
            if let lastWordEnd = getLastWordEnd(currentSegments), lastWordEnd > timeOffset {
              let remainingDuration = windowEndTime - lastWordEnd
              if remainingDuration > threshold {
                seek = Int(lastWordEnd * Float(framesPerSecond))
              } else {
                seek = previousSeek + segmentSize
              }
            }
          }

          // Check first segment for leading silence hallucination
          if let firstSegment = currentSegments.first(where: { $0.words != nil && !$0.words!.isEmpty }) {
            let wordTimings = firstSegment.words!.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }
            if isSegmentAnomaly(wordTimings) {
              let gap = Float(firstSegment.start) - timeOffset
              if gap > threshold {
                seek = previousSeek + Int(gap * Float(framesPerSecond))
                continue
              }
            }
          }

          // Check for hallucinations surrounded by silence
          var halLastEnd = lastSpeechTimestamp
          for si in 0 ..< currentSegments.count {
            let segment = currentSegments[si]
            guard let words = segment.words, !words.isEmpty else { continue }

            let wordTimings = words.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }

            if isSegmentAnomaly(wordTimings) {
              let segmentStart = Float(segment.start)
              let segmentEnd = Float(segment.end)

              // Find next segment with words
              let nextSeg = currentSegments[(si + 1)...].first { $0.words != nil && !$0.words!.isEmpty }
              let halNextStart: Float = if let next = nextSeg, let firstWord = next.words?.first {
                Float(firstWord.start)
              } else {
                timeOffset + segmentDuration
              }

              let silenceBefore = (segmentStart - halLastEnd > threshold) ||
                (segmentStart < threshold) ||
                (segmentStart - timeOffset < 2.0)

              let nextWordTimings: [WordTiming]? = nextSeg?.words?.map {
                WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
              }
              // Python: window_end_time - segment["end"] < 2.0
              let silenceAfter = (halNextStart - segmentEnd > threshold) ||
                isSegmentAnomaly(nextWordTimings) ||
                (windowEndTime - segmentEnd < 2.0)

              if silenceBefore, silenceAfter {
                seek = Int(max(timeOffset + 1, segmentStart) * Float(framesPerSecond))
                if contentDuration - segmentEnd < threshold {
                  seek = contentFrames
                }
                currentSegments.removeSubrange(si...)
                break
              }
            }
            halLastEnd = Float(segment.end)
          }
        }

        // Update last speech timestamp (outside hallucination check, inside word_timestamps)
        // Python lines 822-824
        if let lastWordEnd = getLastWordEnd(currentSegments) {
          lastSpeechTimestamp = lastWordEnd
        }
      }

      // Clear empty segments (matches Python: keep segment but clear text/tokens/words)
      // Python lines 836-844: if start == end or text.strip() == "", clear fields but keep segment
      currentSegments = currentSegments.map { segment in
        if segment.start == segment.end || segment.text.trimmingCharacters(in: .whitespaces).isEmpty {
          return TranscriptionSegment(
            text: "",
            start: segment.start,
            end: segment.end,
            tokens: [],
            avgLogProb: segment.avgLogProb,
            noSpeechProb: segment.noSpeechProb,
            words: []
          )
        }
        return segment
      }

      // Add segments and tokens
      allSegments.append(contentsOf: currentSegments)
      for segment in currentSegments {
        allTokens.append(contentsOf: segment.tokens)
      }

      // Reset prompt if temperature was high (use actual decode temperature, not parameter)
      // Python: if not condition_on_previous_text or result.temperature > 0.5
      if !conditionOnPreviousText || result.temperature > 0.5 {
        promptResetSince = allTokens.count
      }
    }

    let audioDuration = Double(audio.shape[0]) / Double(sampleRate)
    let fullText = allSegments.map { $0.text }.joined(separator: " ").trimmingCharacters(in: .whitespaces)

    let transcribeEndTime = CFAbsoluteTimeGetCurrent()
    let totalTime = transcribeEndTime - transcribeStartTime

    Log.model.info("Transcription complete: \(String(format: "%.2f", totalTime))s for \(String(format: "%.2f", audioDuration))s audio (RTF: \(String(format: "%.2f", totalTime / audioDuration)))")

    return TranscriptionResult(
      text: fullText,
      language: detectedLanguage ?? language ?? "en",
      segments: allSegments,
      processingTime: totalTime,
      duration: audioDuration
    )
  }

  /// Pad or trim mel spectrogram to specified length
  private func padOrTrimMel(_ mel: MLXArray, length: Int) -> MLXArray {
    let currentLength = mel.shape[0]
    if currentLength == length {
      return mel
    } else if currentLength > length {
      return mel[0 ..< length]
    } else {
      // Pad with zeros
      let padding = MLXArray.zeros([length - currentLength, mel.shape[1]])
      return MLX.concatenated([mel, padding], axis: 0)
    }
  }

  // MARK: - Language Detection

  /// Detect the language of audio
  ///
  /// - Parameter audio: Audio waveform (T,) in 16 kHz
  /// - Returns: Tuple of (language_code, probability)
  func detectLanguage(audio: MLXArray) -> (String, Float) {
    // Pad or trim to 30 seconds
    let paddedAudio = padOrTrim(audio)
    eval(paddedAudio)

    // Compute mel spectrogram
    let mel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    let melTransposed = mel.transposed()
    // Ensure exactly 3000 frames to match encoder expectations
    let melTrimmed = padOrTrimMel(melTransposed, length: WhisperAudio.nFrames)
    let batchedMel = melTrimmed.expandedDimensions(axis: 0)

    return detectLanguageFromMel(batchedMel)
  }

  /// Detect language from mel spectrogram
  ///
  /// - Parameter mel: Mel spectrogram (batch=1 or unbatched)
  /// - Returns: Tuple of (language_code, probability)
  private func detectLanguageFromMel(_ mel: MLXArray) -> (String, Float) {
    // Add batch dimension if needed
    var melBatched = mel
    if mel.ndim == 2 {
      melBatched = mel.expandedDimensions(axis: 0)
    }

    // Encode audio
    let audioFeatures = model.encode(melBatched)

    // Create SOT token
    let sotToken = MLXArray([Int32(tokenizer.sot)]).expandedDimensions(axis: 0)

    // Get logits for first token after SOT
    let (logits, _, _) = model.decode(sotToken, audioFeatures: audioFeatures)

    // Extract language token logits
    // Language tokens start at sot + 1 and span 99 tokens
    // Multilingual: 50259-50357, English-only: 50258-50356
    let languageTokenStart = tokenizer.sot + 1
    let languageTokenEnd = tokenizer.sot + 100 // Exclusive (99 language tokens)
    let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]

    // Find language with highest probability
    let probs = MLX.softmax(languageLogits, axis: -1)
    let maxIdx = MLX.argMax(probs).item(Int32.self)
    let maxProb = probs[Int(maxIdx)].item(Float.self)

    // Map index to language code
    let languageIdx = Int(maxIdx)
    let languageCode = Self.languageCodes[languageIdx] ?? "en"

    return (languageCode, maxProb)
  }

  /// Language codes indexed by position (token offset from 50259)
  private static let languageCodes: [Int: String] = [
    0: "en", 1: "zh", 2: "de", 3: "es", 4: "ru", 5: "ko",
    6: "fr", 7: "ja", 8: "pt", 9: "tr", 10: "pl", 11: "ca",
    12: "nl", 13: "ar", 14: "sv", 15: "it", 16: "id", 17: "hi",
    18: "fi", 19: "vi", 20: "he", 21: "uk", 22: "el", 23: "ms",
    24: "cs", 25: "ro", 26: "da", 27: "hu", 28: "ta", 29: "no",
    30: "th", 31: "ur", 32: "hr", 33: "bg", 34: "lt", 35: "la",
    36: "mi", 37: "ml", 38: "cy", 39: "sk", 40: "te", 41: "fa",
    42: "lv", 43: "bn", 44: "sr", 45: "az", 46: "sl", 47: "kn",
    48: "et", 49: "mk", 50: "br", 51: "eu", 52: "is", 53: "hy",
    54: "ne", 55: "mn", 56: "bs", 57: "kk", 58: "sq", 59: "sw",
    60: "gl", 61: "mr", 62: "pa", 63: "si", 64: "km", 65: "sn",
    66: "yo", 67: "so", 68: "af", 69: "oc", 70: "ka", 71: "be",
    72: "tg", 73: "sd", 74: "gu", 75: "am", 76: "yi", 77: "lo",
    78: "uz", 79: "fo", 80: "ht", 81: "ps", 82: "tk", 83: "nn",
    84: "mt", 85: "sa", 86: "lb", 87: "my", 88: "bo", 89: "tl",
    90: "mg", 91: "as", 92: "tt", 93: "haw", 94: "ln", 95: "ha",
    96: "ba", 97: "jw", 98: "su",
  ]

  // MARK: - Audio Segmentation

  /// Segment long audio into 30-second chunks
  ///
  /// - Parameter audio: Audio waveform (T,)
  /// - Returns: Array of audio segments
  private func segmentAudio(_ audio: MLXArray) -> [MLXArray] {
    let audioLength = audio.shape[0]
    let chunkSamples = WhisperAudio.nSamples // 480,000 samples (30s at 16kHz)

    // If audio is shorter than or equal to 30 seconds, return as single segment
    if audioLength <= chunkSamples {
      return [audio]
    }

    // Split into 30-second chunks
    var segments: [MLXArray] = []
    var start = 0

    while start < audioLength {
      let end = min(start + chunkSamples, audioLength)
      let segment = audio[start ..< end]
      segments.append(segment)
      start = end
    }

    return segments
  }
}
