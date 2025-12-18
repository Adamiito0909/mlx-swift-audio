// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import Tokenizers

/// Fun-ASR tokenizer wrapper for Qwen3
///
/// Uses swift-transformers AutoTokenizer for Qwen3 tokenization.
/// Tracks special token IDs for speech recognition prompts.
class FunASRTokenizer {
  private let tokenizer: Tokenizer

  // Special token IDs
  let sosTokenId: Int?
  let eosTokenId: Int?
  let eosTokenIds: Set<Int>
  let imStartTokenId: Int?
  let imEndTokenId: Int?

  // Configuration
  let config: FunASRConfig

  private init(tokenizer: Tokenizer, config: FunASRConfig) {
    self.tokenizer = tokenizer
    self.config = config

    // Resolve special token IDs
    // Try to encode special tokens to get their IDs
    sosTokenId = Self.resolveTokenId(tokenizer: tokenizer, token: config.sosToken)
    eosTokenId = Self.resolveTokenId(tokenizer: tokenizer, token: config.eosToken)
    imStartTokenId = Self.resolveTokenId(tokenizer: tokenizer, token: config.imStartToken)
    imEndTokenId = Self.resolveTokenId(tokenizer: tokenizer, token: config.imEndToken)

    // Build set of EOS token IDs for stopping generation
    var eosIds = Set<Int>()
    if let eosId = eosTokenId {
      eosIds.insert(eosId)
    }
    // Also add common EOS tokens
    let commonEosTokens = ["<|endoftext|>", "<|im_end|>", "</s>"]
    for token in commonEosTokens {
      if let tokenId = Self.resolveTokenId(tokenizer: tokenizer, token: token) {
        eosIds.insert(tokenId)
      }
    }
    // Try to get the tokenizer's EOS token
    if let tokenizerEos = tokenizer.eosTokenId {
      eosIds.insert(tokenizerEos)
    }
    eosTokenIds = eosIds
  }

  /// Resolve a token to its ID
  ///
  /// Takes the first token ID from encoding, similar to Python's approach.
  /// This handles both single-token and multi-token encodings.
  private static func resolveTokenId(tokenizer: Tokenizer, token: String) -> Int? {
    let encoded = tokenizer.encode(text: token)
    // Take the first token ID if available
    return encoded.first
  }

  /// Load tokenizer from model directory
  ///
  /// - Parameters:
  ///   - modelDirectory: Path to model directory containing tokenizer files
  ///   - config: Fun-ASR configuration
  /// - Returns: Initialized tokenizer
  static func load(from modelDirectory: URL, config: FunASRConfig) async throws -> FunASRTokenizer {
    // Load tokenizer from the model directory
    let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
    return FunASRTokenizer(tokenizer: tokenizer, config: config)
  }

  /// Load tokenizer from HuggingFace Hub
  ///
  /// - Parameters:
  ///   - repoId: HuggingFace repository ID
  ///   - config: Fun-ASR configuration
  /// - Returns: Initialized tokenizer
  static func load(from repoId: String, config: FunASRConfig) async throws -> FunASRTokenizer {
    let tokenizer = try await AutoTokenizer.from(pretrained: repoId)
    return FunASRTokenizer(tokenizer: tokenizer, config: config)
  }

  /// Encode text to token IDs
  ///
  /// - Parameter text: Text to encode
  /// - Returns: Array of token IDs
  func encode(_ text: String) -> [Int] {
    tokenizer.encode(text: text)
  }

  /// Decode token IDs to text
  ///
  /// - Parameter tokens: Array of token IDs
  /// - Returns: Decoded text
  func decode(_ tokens: [Int]) -> String {
    tokenizer.decode(tokens: tokens)
  }

  /// Check if a token ID is an EOS token
  ///
  /// - Parameter tokenId: Token ID to check
  /// - Returns: True if the token is an EOS token
  func isEosToken(_ tokenId: Int) -> Bool {
    eosTokenIds.contains(tokenId)
  }

  /// Build the prompt template for transcription/translation
  ///
  /// The template follows the Qwen3 chat format:
  /// ```
  /// <|im_start|>system
  /// {system_prompt}<|im_end|>
  /// <|im_start|>user
  /// <|startofspeech|><|endofspeech|><|im_end|>
  /// <|im_start|>assistant
  /// ```
  ///
  /// - Parameters:
  ///   - task: Task type (transcribe or translate)
  ///   - language: Source language (or "auto" for detection)
  ///   - targetLanguage: Target language for translation
  ///   - initialPrompt: Custom instructions to include
  /// - Returns: Encoded token IDs for the prompt
  func buildPrompt(
    task: FunASRTask,
    language: FunASRLanguage = .auto,
    targetLanguage: FunASRLanguage = .english,
    initialPrompt: String? = nil
  ) -> [Int] {
    let systemPrompt = buildSystemPrompt(
      task: task,
      language: language,
      targetLanguage: targetLanguage,
      initialPrompt: initialPrompt
    )

    let promptParts = [
      "\(config.imStartToken)system\n\(systemPrompt)\(config.imEndToken)",
      "\(config.imStartToken)user\n",
      "\(config.sosToken)\(config.eosToken)",
      "\(config.imEndToken)",
      "\(config.imStartToken)assistant\n",
    ]
    let prompt = promptParts.joined()

    return encode(prompt)
  }

  /// Build system prompt based on task and language settings
  ///
  /// - Parameters:
  ///   - task: Task type
  ///   - language: Source language
  ///   - targetLanguage: Target language for translation
  ///   - initialPrompt: Custom instructions
  /// - Returns: System prompt string
  func buildSystemPrompt(
    task: FunASRTask,
    language: FunASRLanguage,
    targetLanguage: FunASRLanguage,
    initialPrompt: String?
  ) -> String {
    let basePrompt: String
    switch task {
      case .translate:
        let targetLangName = targetLanguage.displayName
        if language == .auto {
          basePrompt =
            "You are a speech translation assistant. Listen to the audio and translate the speech into \(targetLangName). Output only the translation, nothing else."
        } else {
          let sourceLangName = language.displayName
          basePrompt =
            "You are a speech translation assistant. The audio is in \(sourceLangName). Translate it into \(targetLangName). Output only the translation, nothing else."
        }
      case .transcribe:
        if language == .auto {
          basePrompt =
            "You are a speech recognition assistant. Transcribe the audio accurately. Output only the transcription, nothing else."
        } else {
          let langName = language.displayName
          basePrompt =
            "You are a speech recognition assistant. The audio is in \(langName). Transcribe it accurately. Output only the transcription, nothing else."
        }
    }

    if let initialPrompt {
      return "\(initialPrompt)\n\n\(basePrompt)"
    }
    return basePrompt
  }

  /// Find the positions of SOS and EOS tokens in token IDs
  ///
  /// - Parameter tokenIds: Array of token IDs
  /// - Returns: Tuple of (sosPosition, eosPosition), nil if not found
  func findSpeechTokenPositions(_ tokenIds: [Int]) -> (sosPosition: Int, eosPosition: Int)? {
    guard let sosId = sosTokenId, let eosId = eosTokenId else {
      return nil
    }

    var sosPos: Int?
    var eosPos: Int?

    for (i, tokenId) in tokenIds.enumerated() {
      if tokenId == sosId, sosPos == nil {
        sosPos = i
      }
      if tokenId == eosId, eosPos == nil {
        eosPos = i
      }
    }

    if let sos = sosPos, let eos = eosPos {
      return (sos, eos)
    }
    return nil
  }

  /// Clean output text by removing special tokens and artifacts
  ///
  /// - Parameter text: Raw generated text
  /// - Returns: Cleaned text
  func cleanOutput(_ text: String) -> String {
    var cleaned = text

    // Remove thinking blocks
    let thinkPattern = #"<think>.*?</think>"#
    if let regex = try? NSRegularExpression(pattern: thinkPattern, options: .dotMatchesLineSeparators) {
      cleaned = regex.stringByReplacingMatches(
        in: cleaned,
        range: NSRange(cleaned.startIndex..., in: cleaned),
        withTemplate: ""
      )
    }

    // Remove special tokens
    let specialTokens = [
      config.imStartToken,
      config.imEndToken,
      config.sosToken,
      config.eosToken,
      "<|endoftext|>",
    ]
    for token in specialTokens {
      cleaned = cleaned.replacingOccurrences(of: token, with: "")
    }

    return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
