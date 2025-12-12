import Foundation
import TiktokenSwift

/// Whisper tokenizer using TiktokenSwift for BPE tokenization
///
/// Provides quick access to special tokens and language-specific encoding.
/// Token IDs differ between multilingual and English-only models:
/// - Multilingual: eot=50257, sot=50258, 99 language tokens, timestamp_begin=50364
/// - English-only: eot=50256, sot=50257, NO language tokens, timestamp_begin=50363
class WhisperTokenizer {
  private let encoding: CoreBpe
  private let specialTokens: [String: Int]
  let isMultilingual: Bool

  // Special token IDs - set based on whether this is multilingual or English-only
  // CRITICAL: These values must match Python mlx-audio tokenizer exactly
  let eot: Int
  let sot: Int
  let translate: Int
  let transcribe: Int
  let sotLm: Int
  let sotPrev: Int
  let noSpeech: Int
  let noTimestamps: Int
  let timestampBegin: Int

  private init(encoding: CoreBpe, specialTokens: [String: Int], isMultilingual: Bool) {
    self.encoding = encoding
    self.specialTokens = specialTokens
    self.isMultilingual = isMultilingual

    // Compute token IDs dynamically to match Python's get_encoding()
    // Both multilingual and English-only have the same structure, just offset by 1
    // due to different base vocab sizes (50257 vs 50256)
    let baseVocabSize = isMultilingual ? 50257 : 50256
    var nextId = baseVocabSize

    eot = nextId; nextId += 1
    sot = nextId; nextId += 1
    // Skip 99 language tokens
    nextId += 99
    translate = nextId; nextId += 1
    transcribe = nextId; nextId += 1
    sotLm = nextId; nextId += 1
    sotPrev = nextId; nextId += 1
    noSpeech = nextId; nextId += 1
    noTimestamps = nextId; nextId += 1
    timestampBegin = nextId
  }

  /// Load tokenizer for Whisper
  ///
  /// Loads the appropriate Whisper tokenizer vocabulary. First checks the model directory
  /// for vocab files (included in mlx-audio-plus converted models), then falls back to
  /// downloading from GitHub.
  ///
  /// - Parameters:
  ///   - isMultilingual: Whether to load multilingual or English-only vocabulary
  ///   - modelDirectory: Optional path to model directory containing vocab files
  /// - Returns: Initialized tokenizer
  static func load(isMultilingual: Bool, modelDirectory: URL? = nil) async throws -> WhisperTokenizer {
    // Whisper has two vocabulary files:
    // 1. multilingual.tiktoken - Used by multilingual Whisper models (tiny, base, small, medium, large-v3, large-v3-turbo)
    //    Contains 50,257 base vocabulary tokens optimized for multilingual speech recognition
    //    and translation across 99 languages
    //
    // 2. gpt2.tiktoken - Used by English-only Whisper models (tiny.en, base.en, small.en, medium.en)
    //    Contains the standard GPT-2 vocabulary (50,256 tokens) for English-only transcription

    let vocabularyFile = isMultilingual ? "multilingual.tiktoken" : "gpt2.tiktoken"

    // First, check if vocab file exists in model directory (mlx-audio-plus converted models)
    var tiktokenFile: URL?
    if let modelDirectory {
      let modelVocabFile = modelDirectory.appending(path: vocabularyFile)
      if FileManager.default.fileExists(atPath: modelVocabFile.path) {
        tiktokenFile = modelVocabFile
        Log.model.info("Using tiktoken vocabulary from model directory")
      }
    }

    // Fall back to downloading from OpenAI GitHub if not found in model directory
    if tiktokenFile == nil {
      Log.model.info("Downloading tiktoken vocabulary from GitHub...")
      let githubURL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/\(vocabularyFile)"

      // Download to cache directory
      let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("mlx-swift-audio/whisper/assets", isDirectory: true)
      try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

      let cachedFile = cacheDir.appendingPathComponent(vocabularyFile)

      // Download if not already cached
      if !FileManager.default.fileExists(atPath: cachedFile.path) {
        guard let url = URL(string: githubURL) else {
          throw STTError.tokenizationFailed("Invalid GitHub URL for \(vocabularyFile)")
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        try data.write(to: cachedFile)
      }

      tiktokenFile = cachedFile
    }

    guard let tiktokenFile, FileManager.default.fileExists(atPath: tiktokenFile.path) else {
      throw STTError.tokenizationFailed("\(vocabularyFile) not found")
    }

    // Load and parse the tiktoken format (base64-encoded tokens with ranks)
    let data = try Data(contentsOf: tiktokenFile)
    let mergeableRanks = try parseTiktokenBpe(data)

    // Build Whisper-specific special tokens (these are added AFTER the base vocab)
    // Token IDs differ between multilingual and English-only models
    let specialTokens = buildSpecialTokens(isMultilingual: isMultilingual)

    // Convert to UInt32 for TiktokenSwift
    let specialTokensUInt32 = specialTokens.mapValues { UInt32($0) }

    // Whisper uses this pattern for tokenization
    let pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    // Create CoreBpe with Whisper vocabulary + special tokens
    let whisperEncoding = try newCoreBpe(
      encoder: mergeableRanks,
      specialTokensEncoder: specialTokensUInt32,
      pattern: pattern
    )

    return WhisperTokenizer(encoding: whisperEncoding, specialTokens: specialTokens, isMultilingual: isMultilingual)
  }

  /// Parse tiktoken BPE format
  ///
  /// Format: base64-encoded token followed by space and rank
  private static func parseTiktokenBpe(_ data: Data) throws -> [[UInt8]: UInt32] {
    guard let content = String(data: data, encoding: .utf8) else {
      throw STTError.tokenizationFailed("Invalid tiktoken data encoding")
    }

    var encoder: [[UInt8]: UInt32] = [:]

    // Split by lines and parse each line
    let lines = content.split(separator: "\n")
    for line in lines {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty { continue }

      // Each line has format: "base64_token rank"
      let parts = trimmed.split(separator: " ", maxSplits: 1)
      guard parts.count == 2,
            let rank = UInt32(parts[1])
      else {
        continue
      }

      // Decode the base64 token
      guard let tokenData = Data(base64Encoded: String(parts[0])) else {
        continue
      }

      // Store as byte array
      encoder[Array(tokenData)] = rank
    }

    return encoder
  }

  /// Build all Whisper special tokens
  ///
  /// Mirrors Python's get_encoding() function. Both multilingual and English-only
  /// tokenizers include 99 language tokens in the vocabulary. The difference is:
  /// - Multilingual (multilingual.tiktoken): base vocab 50257, starts special at 50257
  /// - English-only (gpt2.tiktoken): base vocab 50256, starts special at 50256
  ///
  /// Special token order (from Python):
  /// endoftext, startoftranscript, [99 language tokens], translate, transcribe,
  /// startoflm, startofprev, nospeech, notimestamps, [1501 timestamp tokens]
  private static func buildSpecialTokens(isMultilingual: Bool) -> [String: Int] {
    var tokens: [String: Int] = [:]

    // Base vocab size determines where special tokens start
    // multilingual.tiktoken: 50257 base tokens
    // gpt2.tiktoken: 50256 base tokens
    let baseVocabSize = isMultilingual ? 50257 : 50256
    var nextTokenId = baseVocabSize

    // Core tokens
    tokens["<|endoftext|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|startoftranscript|>"] = nextTokenId
    nextTokenId += 1

    // Language tokens: <|en|>, <|zh|>, etc. (99 languages)
    // Both multilingual and English-only tokenizers include these in the vocab
    let languageCodes = [
      "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
      "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
      "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
      "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
      "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
      "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
      "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
      "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
      "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
      "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ]

    for lang in languageCodes {
      tokens["<|\(lang)|>"] = nextTokenId
      nextTokenId += 1
    }

    // Task and control tokens
    tokens["<|translate|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|transcribe|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|startoflm|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|startofprev|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|nospeech|>"] = nextTokenId
    nextTokenId += 1
    tokens["<|notimestamps|>"] = nextTokenId
    nextTokenId += 1

    // Timestamp tokens: <|0.00|> through <|30.00|> (1501 tokens, 0.02s increments)
    for i in 0 ... 1500 {
      let seconds = Float(i) * 0.02
      tokens["<|\(String(format: "%.2f", seconds))|>"] = nextTokenId
      nextTokenId += 1
    }

    return tokens
  }

  /// Encode text to token IDs
  ///
  /// - Parameter text: Input text
  /// - Returns: Token IDs
  func encode(_ text: String) -> [Int] {
    // Use encodeOrdinary to avoid encoding special tokens in regular text
    encoding.encodeOrdinary(text: text).map { Int($0) }
  }

  /// Encode text with special tokens
  ///
  /// - Parameter text: Input text that may contain special tokens
  /// - Returns: Token IDs
  func encodeWithSpecialTokens(_ text: String) -> [Int] {
    encoding.encodeWithSpecialTokens(text: text).map { Int($0) }
  }

  /// Decode token IDs to text
  ///
  /// Filters out timestamp tokens (special tokens in [eot, timestampBegin) are kept
  /// but won't appear in normal transcription output anyway)
  ///
  /// - Parameter tokens: Token IDs
  /// - Returns: Decoded text
  func decode(_ tokens: [Int]) -> String {
    // Filter out timestamp tokens (matches Python: t < timestamp_begin)
    // Keeps base vocab (0 to eot-1) and special tokens (eot to timestampBegin-1)
    // In practice, special tokens don't appear in transcription output
    let textTokens = tokens.filter { $0 < timestampBegin }

    // Decode using CoreBpe
    let tokensUInt32 = textTokens.map { UInt32($0) }
    return (try? encoding.decode(tokens: tokensUInt32)) ?? ""
  }

  /// Decode token IDs including timestamps
  ///
  /// - Parameter tokens: Token IDs
  /// - Returns: Decoded text with timestamp annotations
  func decodeWithTimestamps(_ tokens: [Int]) -> String {
    var result = ""
    var currentText = ""
    var currentTokens: [UInt32] = []

    for token in tokens {
      if isTimestampToken(token) {
        // Flush any accumulated text
        if !currentTokens.isEmpty {
          if let decoded = try? encoding.decode(tokens: currentTokens) {
            currentText += decoded
          }
          currentTokens.removeAll()
        }

        // Add timestamp annotation
        let time = decodeTimestamp(token)
        result += currentText
        result += "<|\(String(format: "%.2f", time))|>"
        currentText = ""
      } else if token < timestampBegin {
        // Text token (base vocab or special tokens that aren't timestamps)
        currentTokens.append(UInt32(token))
      }
      // else: skip - shouldn't happen (timestamps handled above)
    }

    // Flush remaining text
    if !currentTokens.isEmpty {
      if let decoded = try? encoding.decode(tokens: currentTokens) {
        result += decoded
      }
    }

    return result
  }

  /// Build SOT (Start-Of-Transcript) sequence
  ///
  /// Mirrors Python's tokenizer.sot_sequence property:
  /// - Multilingual models: [sot, language_token, task_token]
  /// - English-only models: [sot] only (no language or task token!)
  ///
  /// English-only models were trained with just [sot] as the prompt prefix.
  /// The model directly predicts timestamps and text after the sot token.
  ///
  /// - Parameters:
  ///   - language: Optional language code (e.g., "en", "zh") - ignored for English-only
  ///   - task: Transcription task - ignored for English-only
  /// - Returns: SOT token sequence
  func sotSequence(language: String?, task: TranscriptionTask) -> [Int] {
    var sequence = [sot]

    // English-only models use only [sot] - no language or task tokens
    // They were trained to directly predict output after just the sot token
    guard isMultilingual else {
      return sequence
    }

    // Multilingual models: add language token if specified
    if let language, let langToken = specialTokens["<|\(language)|>"] {
      sequence.append(langToken)
    }

    // Multilingual models: add task token
    let taskToken = task == .transcribe ? transcribe : translate
    sequence.append(taskToken)

    return sequence
  }

  /// Build SOT sequence including no-timestamps token
  ///
  /// - Parameters:
  ///   - language: Optional language code
  ///   - task: Transcription task
  /// - Returns: SOT token sequence with no-timestamps
  func sotSequenceIncludingNoTimestamps(language: String?, task: TranscriptionTask) -> [Int] {
    var sequence = sotSequence(language: language, task: task)
    sequence.append(noTimestamps)
    return sequence
  }

  /// Get language token for a language code
  ///
  /// - Parameter language: Language code (e.g., "en")
  /// - Returns: Language token ID, or nil if not found
  func languageToken(for language: String) -> Int? {
    specialTokens["<|\(language)|>"]
  }

  /// Check if a token is a language token
  ///
  /// - Parameter token: Token ID
  /// - Returns: True if token is a language token
  func isLanguageToken(_ token: Int) -> Bool {
    // Language tokens start at sot + 1 and span 99 tokens
    let languageTokenStart = sot + 1
    let languageTokenEnd = sot + 100 // Exclusive (99 language tokens)
    return token >= languageTokenStart && token < languageTokenEnd
  }

  /// Check if a token is a timestamp token
  ///
  /// - Parameter token: Token ID
  /// - Returns: True if token is a timestamp token
  func isTimestampToken(_ token: Int) -> Bool {
    token >= timestampBegin
  }

  /// Decode timestamp token to seconds
  ///
  /// - Parameter token: Timestamp token ID
  /// - Returns: Time in seconds
  func decodeTimestamp(_ token: Int) -> Float {
    guard token >= timestampBegin else { return 0.0 }
    let index = token - timestampBegin
    return Float(index) * 0.02
  }

  /// Get list of non-speech tokens to suppress
  ///
  /// Returns tokens for speaker tags and non-speech annotations like:
  /// - ♪♪♪ (music)
  /// - ( SPEAKING FOREIGN LANGUAGE )
  /// - [DAVID] (speaker names)
  ///
  /// - Returns: Array of token IDs to suppress
  func nonSpeechTokens() -> [Int] {
    // Symbols to suppress (split into individual characters and strings)
    var symbols: [String] = []

    // Individual characters
    for char in "\"#()*+/:;<=>@[\\]^_`{|}~「」『』" {
      symbols.append(String(char))
    }

    // Multi-character sequences
    symbols += ["<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪"]

    // Musical symbols (U+2640 to U+267F)
    let miscellaneous = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"]

    var result = Set<Int>()

    // Allow hyphens and quotes between words, but not at the beginning
    let dashTokens = encoding.encodeOrdinary(text: " -").map { Int($0) }
    if !dashTokens.isEmpty {
      result.insert(dashTokens[0])
    }
    let quoteTokens = encoding.encodeOrdinary(text: " '").map { Int($0) }
    if !quoteTokens.isEmpty {
      result.insert(quoteTokens[0])
    }

    // Encode symbols and add their tokens
    for symbol in symbols + miscellaneous {
      // Try encoding the symbol alone
      let tokens = encoding.encodeOrdinary(text: symbol).map { Int($0) }
      if !tokens.isEmpty {
        if tokens.count == 1 || miscellaneous.contains(symbol) {
          result.insert(tokens[0])
        }
      }

      // Try encoding with leading space
      let spacedTokens = encoding.encodeOrdinary(text: " " + symbol).map { Int($0) }
      if !spacedTokens.isEmpty {
        if spacedTokens.count == 1 || miscellaneous.contains(symbol) {
          result.insert(spacedTokens[0])
        }
      }
    }

    return Array(result).sorted()
  }

  // MARK: - Word-Level Timestamp Support

  /// Split tokens into word-level groups for word timestamp alignment
  ///
  /// This method implements language-aware word splitting:
  /// - For CJK languages (Chinese, Japanese, Korean): character-level splitting
  /// - For other languages (Latin, Cyrillic, etc.): whitespace-based splitting
  ///
  /// - Parameter tokens: Token IDs (may include EOT at end)
  /// - Returns: Tuple of (words, token groups per word)
  func splitToWordTokens(_ tokens: [Int]) -> (words: [String], tokenGroups: [[Int]]) {
    // Filter to only text tokens
    let textTokens = tokens.filter { $0 < eot }
    guard !textTokens.isEmpty else {
      // Return EOT marker if present
      if tokens.last == eot {
        return ([""], [[eot]])
      }
      return ([], [])
    }

    // Decode to check if this is CJK-heavy text
    let decoded = decode(textTokens)
    let isCJK = decoded.unicodeScalars.filter { isCJKCharacter($0) }.count > decoded.count / 2

    if isCJK {
      return splitCJK(tokens)
    } else {
      return splitByWhitespace(tokens)
    }
  }

  /// Check if a Unicode scalar is a CJK character
  @inline(__always)
  private func isCJKCharacter(_ scalar: Unicode.Scalar) -> Bool {
    let value = scalar.value
    return (0x4E00 ... 0x9FFF).contains(value) || // CJK Unified Ideographs
      (0x3400 ... 0x4DBF).contains(value) || // CJK Extension A
      (0x20000 ... 0x2A6DF).contains(value) || // CJK Extension B
      (0x3040 ... 0x309F).contains(value) || // Hiragana
      (0x30A0 ... 0x30FF).contains(value) || // Katakana
      (0xAC00 ... 0xD7AF).contains(value) // Korean Hangul
  }

  /// Split CJK text character by character
  ///
  /// For CJK languages, each character is typically a meaningful unit,
  /// so we split at the character level for more accurate timestamps.
  private func splitCJK(_ tokens: [Int]) -> (words: [String], tokenGroups: [[Int]]) {
    var words: [String] = []
    var tokenGroups: [[Int]] = []

    for token in tokens {
      if token >= eot {
        // Add EOT as final word marker
        words.append("")
        tokenGroups.append([token])
        continue
      }

      let decoded = decode([token])
      // Each character becomes a "word" for CJK
      // But we keep the whole token together since BPE may combine chars
      if !decoded.isEmpty {
        words.append(decoded)
        tokenGroups.append([token])
      }
    }

    return (words, tokenGroups)
  }

  /// Split by whitespace for Latin, Cyrillic, and similar scripts
  ///
  /// Whisper's BPE tokenizer typically adds a leading space to the first
  /// token of each word (except the first word), which we use as word boundaries.
  private func splitByWhitespace(_ tokens: [Int]) -> (words: [String], tokenGroups: [[Int]]) {
    var words: [String] = []
    var tokenGroups: [[Int]] = []

    var currentWord = ""
    var currentTokens: [Int] = []

    for token in tokens {
      if token >= eot {
        // End current word if any
        if !currentWord.isEmpty || !currentTokens.isEmpty {
          words.append(currentWord)
          tokenGroups.append(currentTokens)
        }
        // Add EOT marker
        words.append("")
        tokenGroups.append([token])
        break
      }

      let decoded = decode([token])

      // Check if this token starts with whitespace (word boundary)
      // BPE tokens for word starts typically begin with a space character
      if decoded.hasPrefix(" "), !currentWord.isEmpty || !currentTokens.isEmpty {
        // Finish previous word
        words.append(currentWord)
        tokenGroups.append(currentTokens)
        // Start new word (include the space in the word text)
        currentWord = decoded
        currentTokens = [token]
      } else {
        // Continue current word
        currentWord += decoded
        currentTokens.append(token)
      }
    }

    // Handle remaining word (if no EOT was encountered)
    if !currentWord.isEmpty || !currentTokens.isEmpty {
      words.append(currentWord)
      tokenGroups.append(currentTokens)
    }

    return (words, tokenGroups)
  }
}
