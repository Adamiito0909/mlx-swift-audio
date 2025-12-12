import Compression
import Foundation
import MLX
import MLXNN

// MARK: - Decoding Options

/// Options for Whisper decoding
struct DecodingOptions {
  /// Task: transcribe or translate to English
  let task: TranscriptionTask

  /// Language code (e.g., "en", "zh"), nil for auto-detect
  let language: String?

  /// Sampling temperature (0.0 for greedy decoding)
  let temperature: Float

  /// Maximum number of tokens to generate
  let maxTokens: Int

  /// Whether to include timestamps
  let timestamps: TimestampGranularity

  /// Prompt tokens from previous segments (for conditioning on previous text)
  /// These are prepended to the SOT sequence
  let prompt: [Int]

  init(
    task: TranscriptionTask = .transcribe,
    language: String? = nil,
    temperature: Float = 0.0,
    maxTokens: Int = 448,
    timestamps: TimestampGranularity = .segment,
    prompt: [Int] = []
  ) {
    self.task = task
    self.language = language
    self.temperature = temperature
    self.maxTokens = maxTokens
    self.timestamps = timestamps
    self.prompt = prompt
  }

  static let `default` = DecodingOptions()
}

// MARK: - Decoding Result

/// Result from decoding a single audio segment
struct DecodingResult {
  /// Generated token sequence
  let tokens: [Int]

  /// Decoded text
  let text: String

  /// Average log probability
  let avgLogProb: Float

  /// No-speech probability (0-1)
  let noSpeechProb: Float

  /// Temperature used
  let temperature: Float

  /// Compression ratio (text length / gzip(text) length)
  let compressionRatio: Float
}

// MARK: - Greedy Decoder

/// Greedy decoder for Whisper
///
/// Implements simple greedy decoding with KV caching
class GreedyDecoder {
  let model: WhisperModel
  let tokenizer: WhisperTokenizer
  let options: DecodingOptions

  init(model: WhisperModel, tokenizer: WhisperTokenizer, options: DecodingOptions) {
    self.model = model
    self.tokenizer = tokenizer
    self.options = options
  }

  /// Decode an audio segment
  ///
  /// - Parameter mel: Mel spectrogram (batch=1, n_mels, n_frames)
  /// - Returns: Decoding result
  func decode(_ mel: MLXArray) -> DecodingResult {
    // Encode audio to features
    let audioFeatures = model.encode(mel)
    eval(audioFeatures) // Ensure audio features are fully computed

    // Verify audio features are valid (not zero/constant)
    let afStd = audioFeatures.variance().sqrt().item(Float.self)
    if afStd < 0.01 {
      Log.model.error("⚠️  Audio features have very low variance!")
    }

    // Build initial token sequence
    // If prompt tokens are provided, prepend them with <|startofprev|>
    // This matches Python's behavior for condition_on_previous_text
    var tokens: [Int] = []

    if !options.prompt.isEmpty {
      // Add <|startofprev|> followed by prompt tokens (previous segment's output)
      tokens.append(tokenizer.sotPrev)
      tokens.append(contentsOf: options.prompt)
    }

    // Track SOT position for no-speech probability extraction (matches Python's sot_index)
    let sotIndex = tokens.count

    // Add SOT sequence (SOT, language, task tokens)
    tokens.append(contentsOf: tokenizer.sotSequence(language: options.language, task: options.task))

    // Only add no-timestamps token if timestamps are disabled
    // When timestamps are enabled, the first timestamp is the first GENERATED token
    if options.timestamps == .none {
      tokens.append(tokenizer.noTimestamps)
    }

    // Calculate how many tokens we can generate
    // We need to account for the initial SOT sequence
    let initialTokenCount = tokens.count
    let maxGenerateTokens = options.maxTokens - initialTokenCount

    // Autoregressive decoding
    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
    var sumLogProb: Float = 0.0
    var tokenCount = 0
    var noSpeechProb: Float = 0.0

    for iteration in 0 ..< maxGenerateTokens {
      // Convert tokens to MLXArray
      // With KV caching: only pass new token(s), not all tokens
      let tokensToProcess: MLXArray
      if kvCache != nil {
        // Pass only the last token (the new one)
        let lastToken = tokens.last!
        tokensToProcess = MLXArray([Int32(lastToken)]).expandedDimensions(axis: 0)
      } else {
        // First iteration: pass all initial tokens (SOT sequence)
        tokensToProcess = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
      }

      // Forward pass through decoder
      let (logits, newCache, _) = model.decode(
        tokensToProcess,
        audioFeatures: audioFeatures,
        kvCache: kvCache
      )
      eval(logits) // Ensure logits are computed before use

      // Compute no-speech probability from first forward pass
      // Extract logits at SOT position (matches Python: pre_logits[:, sot_index])
      if iteration == 0 {
        // logits shape: [batch=1, seq_len, vocab_size]
        // Use sotIndex to get logits at the SOT position (not the last position)
        // This is where the model decides if there's speech or silence
        let sotLogits = logits[0, sotIndex] // [vocab_size]

        // Apply softmax to get probabilities
        let probs = MLX.softmax(sotLogits, axis: -1)

        // Extract probability for no-speech token
        noSpeechProb = probs[tokenizer.noSpeech].item(Float.self)
      }

      // Update KV cache
      kvCache = newCache

      // Get logits for last token
      var lastLogits = logits[0, -1]
      let vocabSize = lastLogits.shape[0]

      // Get number of generated tokens (excluding initial SOT sequence)
      let numGenerated = tokens.count - initialTokenCount

      // =============================================================================
      // STEP 1: Build base suppression mask (SuppressBlank + SuppressTokens)
      // Uses MLX operations for efficiency instead of Swift loops
      // =============================================================================

      // Create indices array for vectorized comparisons
      let indices = MLXArray(Int32(0) ..< Int32(vocabSize))

      // SuppressTokens: create mask for specific token IDs (small list, loop is fine)
      var suppressTokenIds = tokenizer.nonSpeechTokens()
      suppressTokenIds.append(contentsOf: [
        tokenizer.transcribe,
        tokenizer.translate,
        tokenizer.sot,
        tokenizer.sotPrev,
        tokenizer.sotLm,
        tokenizer.noSpeech,
      ])

      // SuppressBlank: add blank tokens and EOT on first iteration only
      if iteration == 0 {
        suppressTokenIds.append(contentsOf: tokenizer.encode(" "))
        suppressTokenIds.append(tokenizer.eot)
      }

      // Build base mask efficiently: create Swift array, set values, convert once
      var baseMaskValues = [Float](repeating: 0.0, count: vocabSize)
      for tokenId in suppressTokenIds where tokenId < vocabSize {
        baseMaskValues[tokenId] = -Float.infinity
      }
      let baseMask = MLXArray(baseMaskValues)

      // =============================================================================
      // STEP 2: Build timestamp rules mask (ApplyTimestampRules internal mask)
      // Uses vectorized MLX operations for range-based suppressions
      // =============================================================================
      var timestampMask = MLXArray.zeros([vocabSize])

      if options.timestamps != .none {
        // Suppress no-timestamps token
        timestampMask = MLX.where(
          indices .== Int32(tokenizer.noTimestamps),
          MLXArray(-Float.infinity),
          timestampMask
        )

        let lastWasTimestamp = numGenerated >= 1 && tokens.last! >= tokenizer.timestampBegin
        let penultimateWasTimestamp = numGenerated < 2 || tokens[tokens.count - 2] >= tokenizer.timestampBegin

        if lastWasTimestamp {
          if penultimateWasTimestamp {
            // Two timestamps in a row: suppress all timestamps (indices >= timestampBegin)
            timestampMask = MLX.where(
              indices .>= Int32(tokenizer.timestampBegin),
              MLXArray(-Float.infinity),
              timestampMask
            )
          } else {
            // Text then timestamp: suppress all text tokens (indices < eot)
            timestampMask = MLX.where(
              indices .< Int32(tokenizer.eot),
              MLXArray(-Float.infinity),
              timestampMask
            )
          }
        }

        // Enforce timestamp monotonicity (Python lines 359-368)
        // Find the last timestamp TOKEN VALUE in the generated sequence
        // Then forbid generating any timestamp smaller than it
        let generatedTokens = tokens.suffix(numGenerated)
        let timestampTokenValues = generatedTokens.compactMap { token -> Int? in
          token > tokenizer.timestampBegin ? token : nil
        }

        if !timestampTokenValues.isEmpty {
          var lastTimestampToken = timestampTokenValues.last!
          // Force nonzero segment length to prevent infinite looping
          if penultimateWasTimestamp {
            lastTimestampToken += 1
          }
          // Suppress all timestamp tokens from timestampBegin up to (but not including) lastTimestampToken
          let lowerCond = indices .>= Int32(tokenizer.timestampBegin)
          let upperCond = indices .< Int32(lastTimestampToken)
          let rangeCond = MLX.logicalAnd(lowerCond, upperCond)
          timestampMask = MLX.where(rangeCond, MLXArray(-Float.infinity), timestampMask)
        }

        // First generated token MUST be a timestamp (Python lines 370-379)
        if numGenerated == 0 {
          // Suppress all non-timestamp tokens (indices < timestampBegin)
          timestampMask = MLX.where(
            indices .< Int32(tokenizer.timestampBegin),
            MLXArray(-Float.infinity),
            timestampMask
          )

          // Apply max_initial_timestamp option
          let maxInitialTimestampIndex = 50
          let lastAllowed = tokenizer.timestampBegin + maxInitialTimestampIndex
          if lastAllowed < vocabSize {
            // Suppress timestamps beyond max initial (indices > lastAllowed)
            timestampMask = MLX.where(
              indices .> Int32(lastAllowed),
              MLXArray(-Float.infinity),
              timestampMask
            )
          }
        }
      }

      // =============================================================================
      // STEP 3: Apply timestamp probability heuristic (Python lines 381-394)
      // Python's ApplyTimestampRules.apply() receives logits already filtered by
      // SuppressBlank and SuppressTokens, so we use logits + baseMask here
      // =============================================================================
      var forceTimestamp = false
      var timestampLogProbSumValue: Float = 0
      var maxTextLogProbValue: Float = 0
      if options.timestamps != .none, numGenerated > 0 {
        // Python: logits passed to ApplyTimestampRules have SuppressBlank + SuppressTokens applied
        // Swift equivalent: lastLogits + baseMask (baseMask contains those suppressions)
        let filteredLogits = lastLogits + baseMask
        let logProbs = filteredLogits - MLX.logSumExp(filteredLogits, axes: [-1], keepDims: true)

        // Compare timestamp vs text token probabilities
        // Python uses logsumexp (combined probability of all timestamps) vs max text token
        let timestampLogProbSum = MLX.logSumExp(logProbs[tokenizer.timestampBegin...], axes: [-1], keepDims: true)
        let maxTextLogProb = logProbs[0 ..< tokenizer.timestampBegin].max(axes: [-1], keepDims: true)

        timestampLogProbSumValue = timestampLogProbSum.item(Float.self)
        maxTextLogProbValue = maxTextLogProb.item(Float.self)

        if timestampLogProbSumValue > maxTextLogProbValue {
          forceTimestamp = true
        }
      }

      // If forcing timestamp, suppress all text tokens in timestamp mask
      if forceTimestamp {
        timestampMask = MLX.where(
          indices .< Int32(tokenizer.timestampBegin),
          MLXArray(-Float.infinity),
          timestampMask
        )
      }

      // =============================================================================
      // STEP 4: Combine masks and apply (vectorized min to merge -inf values)
      // =============================================================================
      let finalMask = MLX.minimum(baseMask, timestampMask)
      lastLogits = lastLogits + finalMask

      // Sample next token
      let nextToken: Int
      if options.temperature == 0.0 {
        // Greedy decoding
        nextToken = Int(MLX.argMax(lastLogits).item(Int32.self))
      } else {
        // Temperature sampling
        let probs = MLX.softmax(lastLogits / options.temperature, axis: -1)
        nextToken = sampleFromDistribution(probs)
      }

      // Track log probability
      let logProbs = MLX.log(MLX.softmax(lastLogits, axis: -1))
      let logProb = logProbs[nextToken].item(Float.self)
      sumLogProb += logProb
      tokenCount += 1

      // Add token to sequence
      tokens.append(nextToken)

      // Check for end-of-text token
      if nextToken == tokenizer.eot {
        break
      }
    }

    // Compute statistics
    let avgLogProb = tokenCount > 0 ? sumLogProb / Float(tokenCount) : 0.0

    // Extract only the generated tokens (exclude prompt and SOT sequence)
    // This matches Python's behavior where result.tokens only contains generated tokens
    var generatedTokens = Array(tokens[initialTokenCount...])

    // Strip EOT from result tokens (matches Python line 669: t[: t.index(tokenizer.eot)])
    // This is important for timestamp detection in seek-based processing
    if let eotIndex = generatedTokens.firstIndex(of: tokenizer.eot) {
      generatedTokens = Array(generatedTokens[..<eotIndex])
    }

    // Decode text from generated tokens only
    let text = tokenizer.decode(generatedTokens)

    // Compute compression ratio using zlib (matches Python implementation)
    // Higher ratio = more repetitive text (hallucination indicator)
    let compressionRatio = computeCompressionRatio(text)

    return DecodingResult(
      tokens: generatedTokens,
      text: text,
      avgLogProb: avgLogProb,
      noSpeechProb: noSpeechProb,
      temperature: options.temperature,
      compressionRatio: compressionRatio
    )
  }

  /// Sample from a probability distribution
  ///
  /// - Parameter probs: Probability distribution
  /// - Returns: Sampled index
  private func sampleFromDistribution(_ probs: MLXArray) -> Int {
    // Simple categorical sampling
    let probsArray = probs.asArray(Float.self)
    let random = Float.random(in: 0 ..< 1)

    var cumsum: Float = 0.0
    for (i, p) in probsArray.enumerated() {
      cumsum += p
      if cumsum >= random {
        return i
      }
    }

    // Fallback to last token (shouldn't happen)
    return probsArray.count - 1
  }
}

// MARK: - Compression Ratio

/// Compute compression ratio using zlib compression
/// Matches Python: len(text_bytes) / len(zlib.compress(text_bytes))
/// Higher ratio indicates more repetitive text (potential hallucination)
///
/// - Parameter text: Text to analyze
/// - Returns: Compression ratio (uncompressed/compressed size)
func computeCompressionRatio(_ text: String) -> Float {
  guard !text.isEmpty else { return 1.0 }

  let textBytes = Array(text.utf8)
  let sourceSize = textBytes.count

  // Use zlib compression (COMPRESSION_ZLIB matches Python's zlib.compress)
  // Allocate destination buffer - worst case is slight expansion
  let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: sourceSize + 512)
  defer { destinationBuffer.deallocate() }

  let compressedSize = textBytes.withUnsafeBufferPointer { sourceBuffer in
    compression_encode_buffer(
      destinationBuffer,
      sourceSize + 512,
      sourceBuffer.baseAddress!,
      sourceSize,
      nil,
      COMPRESSION_ZLIB
    )
  }

  // Handle compression failure
  guard compressedSize > 0 else { return 1.0 }

  return Float(sourceSize) / Float(compressedSize)
}
