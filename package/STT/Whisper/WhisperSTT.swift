// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

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
    // Multilingual: eot=50257, sot=50258, transcribe=50360, timestamp_begin=50365
    // English-only: eot=50256, sot=50257, transcribe=50359, timestamp_begin=50364
    if model.isMultilingual {
      assert(tokenizer.eot == 50257, "Multilingual EOT token must be 50257")
      assert(tokenizer.sot == 50258, "Multilingual SOT token must be 50258")
      assert(tokenizer.transcribe == 50360, "Multilingual transcribe token must be 50360")
      assert(tokenizer.timestampBegin == 50365, "Multilingual timestamp_begin must be 50365")
    } else {
      assert(tokenizer.eot == 50256, "English-only EOT token must be 50256")
      assert(tokenizer.sot == 50257, "English-only SOT token must be 50257")
      assert(tokenizer.transcribe == 50359, "English-only transcribe token must be 50359")
      assert(tokenizer.timestampBegin == 50364, "English-only timestamp_begin must be 50364")
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
    // Returns (n_frames, n_mels) - already in the right shape for Conv1d
    let fullMel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    eval(fullMel)

    // Content frames (excluding padding)
    let contentFrames = audio.shape[0] / hopLength
    let contentDuration = Float(contentFrames * hopLength) / Float(sampleRate)

    Log.model.info("Transcribing \(String(format: "%.1f", contentDuration))s audio with seek-based processing")

    // Detect language if not specified
    var detectedLanguage: String? = nil
    if language == nil {
      let melSegment = padOrTrimMel(fullMel[0 ..< nFrames], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0).asType(.float16)
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

      Log.model.debug("Processing segment: seek=\(seek) (\(String(format: "%.2f", timeOffset))s), size=\(segmentSize) frames (\(String(format: "%.2f", segmentDuration))s)")

      // Extract mel segment and pad to nFrames
      // Cast to float16 to match Python's behavior (line 612-614 in whisper.py)
      let melSegment = padOrTrimMel(fullMel[seek ..< (seek + segmentSize)], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0).asType(.float16)

      // Build prompt from previous tokens (if conditioning enabled)
      // Use tokens since last prompt reset (matches Python: all_tokens[prompt_reset_since:])
      let promptTokens = conditionOnPreviousText ? Array(allTokens[promptResetSince...]) : []
      let prompt = promptTokens

      // Temperature fallback loop (matches Python's decode_with_fallback)
      // Try increasing temperatures when output is too repetitive (high compression ratio)
      // or has low confidence (low avg_logprob)
      //
      // Optimization: Use fewer temperature steps for very short segments where
      // fallbacks are unlikely to help and just waste time
      let temperatureFallbackSequence: [Float] = segmentDuration < 2.0
        ? [0.0, 0.5, 1.0] // Short segments: 3 steps
        : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] // Normal segments: 6 steps
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

      let decodedText = tokenizer.decode(result.tokens.filter { $0 < tokenizer.eot })
      // Log timestamp tokens for debugging
      let tsTokens = result.tokens.filter { $0 >= tokenizer.timestampBegin }
      let tsPositions = tsTokens.map { Float($0 - tokenizer.timestampBegin) * 0.02 }
      Log.model.debug("Decoded: noSpeechProb=\(String(format: "%.3f", result.noSpeechProb)), avgLogProb=\(String(format: "%.3f", result.avgLogProb)), tokens=\(result.tokens.count), timestamps=\(tsPositions), text='\(decodedText.prefix(100))'")

      // No-speech detection: skip if no_speech_prob > threshold
      if let nsThreshold = noSpeechThreshold {
        var shouldSkip = result.noSpeechProb > nsThreshold

        // Don't skip if logprob is high enough despite high no_speech_prob
        if let lpThreshold = logprobThreshold, result.avgLogProb > lpThreshold {
          shouldSkip = false
        }

        if shouldSkip {
          Log.model.debug("Skipping segment due to no-speech detection (prob=\(String(format: "%.3f", result.noSpeechProb)) > \(nsThreshold))")
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
          // Sanity check: don't let hallucinated timestamps cause seek to jump beyond segment.
          // The model can hallucinate timestamps pointing far into the future (e.g., 25s when
          // only 2s of audio remains), which would cause seek to jump past the end of content.
          let maxSeekAdvance = segmentSize
          let timestampSeek = lastTimestampPos * inputStride
          seek += min(timestampSeek, maxSeekAdvance)
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

      // Ensure seek never moves backward (WhisperKit safety mechanism)
      seek = max(previousSeek, seek)

      Log.model.debug("Seek advanced: \(previousSeek) -> \(seek) (\(String(format: "%.2f", Float(seek * hopLength) / Float(sampleRate)))s), segments=\(currentSegments.count)")

      // Filter out zero-length segments (WhisperKit approach)
      currentSegments = currentSegments.filter { $0.end > $0.start }

      // Filter out segments with timestamps that exceed the segment window
      // This catches hallucinations where model generates impossible timestamps (e.g., 20s in a 2s segment)
      currentSegments = currentSegments.filter { segment in
        let relativeEnd = Float(segment.end) - timeOffset
        if relativeEnd > segmentDuration + 1.0 { // Allow 1s tolerance
          Log.model.warning("Filtering hallucinated segment (timestamp \(String(format: "%.1f", relativeEnd))s exceeds \(String(format: "%.1f", segmentDuration))s window): '\(segment.text.prefix(50))'")
          return false
        }
        return true
      }

      // Filter segments with very low confidence after temperature exhaustion
      // These are likely hallucinations from silence/unclear audio at end of content
      if result.temperature >= 0.8, result.avgLogProb < -2.0 {
        currentSegments = currentSegments.filter { segment in
          let trimmedText = segment.text.trimmingCharacters(in: .whitespaces)
          if !trimmedText.isEmpty {
            Log.model.warning("Filtering low-confidence segment (avgLogProb=\(String(format: "%.2f", result.avgLogProb)), temp=\(result.temperature)): '\(trimmedText.prefix(50))'")
          }
          return false
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

      // Filter out problematic segments (inspired by WhisperKit's threshold-based approach):
      // 1. Zero-duration segments (start == end)
      // 2. Empty text after trimming whitespace
      // 3. Punctuation-only segments (likely artifacts)
      // 4. High no-speech probability segments (likely silence/hallucination)
      // 5. For word timestamps: segments where word alignment failed but text exists
      let punctuationOnly = CharacterSet.punctuationCharacters.union(.whitespaces)
      currentSegments = currentSegments.filter { segment in
        // Keep segments with valid duration and non-empty meaningful text
        let trimmedText = segment.text.trimmingCharacters(in: .whitespaces)
        let hasMeaningfulText = !trimmedText.isEmpty &&
          !trimmedText.unicodeScalars.allSatisfy { punctuationOnly.contains($0) }

        // Filter high no-speech probability segments (WhisperKit uses 0.6 default threshold)
        // These are likely hallucinations from silence/padding
        if segment.noSpeechProb > 0.9 {
          Log.model.warning("Filtering high no-speech segment (\(segment.noSpeechProb)): '\(trimmedText.prefix(50))'")
          return false
        }

        // For word timestamps: apply multiple hallucination checks
        if timestamps == .word {
          let hasWords = segment.words != nil && !segment.words!.isEmpty
          // If we have text but no words, be suspicious - only keep if it's very short text
          if hasMeaningfulText, !hasWords, trimmedText.count > 10 {
            Log.model.warning("Filtering potential hallucination (no word alignment): '\(trimmedText.prefix(50))'")
            return false
          }

          // Check for anomalous word patterns (very short, very long, or low probability)
          // This catches hallucinations like "iğ", "B gensham", etc.
          if hasWords {
            let wordTimings = segment.words!.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }
            if isSegmentAnomaly(wordTimings) {
              Log.model.warning("Filtering anomalous segment (suspicious word patterns): '\(trimmedText.prefix(50))'")
              return false
            }
          }
        }

        return segment.start != segment.end && hasMeaningfulText
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
    // Decode all tokens together to preserve natural spacing (matching WhisperKit/Python behavior)
    // Filter out special tokens (timestamps, etc.) before decoding
    // This avoids double spaces that occur when joining segment texts with a separator
    let textTokens = allTokens.filter { $0 < tokenizer.eot }
    let fullText = tokenizer.decode(textTokens).trimmingCharacters(in: .whitespaces)

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

    // Compute mel spectrogram - returns (n_frames, n_mels)
    let mel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    // Ensure exactly 3000 frames to match encoder expectations
    let melTrimmed = padOrTrimMel(mel, length: WhisperAudio.nFrames)
    // Cast to float16 to match Python's behavior
    let batchedMel = melTrimmed.expandedDimensions(axis: 0).asType(.float16)

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
    // Language tokens start at sot + 1 and span WHISPER_NUM_LANGUAGES (100) tokens
    // Multilingual: 50259-50358, English-only: 50258-50357
    let languageTokenStart = tokenizer.sot + 1
    let languageTokenEnd = tokenizer.sot + 1 + WHISPER_NUM_LANGUAGES // Exclusive
    let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]

    // Find language with highest probability
    let probs = MLX.softmax(languageLogits, axis: -1)
    let maxIdx = MLX.argMax(probs).item(Int32.self)
    let maxProb = probs[Int(maxIdx)].item(Float.self)

    // Map index to language code using tokenizer's single source of truth
    let languageIdx = Int(maxIdx)
    let languageCode = tokenizer.languageCode(forIndex: languageIdx) ?? "en"

    return (languageCode, maxProb)
  }

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
