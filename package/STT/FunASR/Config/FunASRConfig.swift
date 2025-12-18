// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation

// MARK: - Model Repository

/// Fun-ASR model type
public enum FunASRModelType: String, CaseIterable, Sendable {
  /// Standard Fun-ASR Nano - optimized for transcription
  case nano = "Fun-ASR-Nano"

  /// Multilingual Fun-ASR (MLT) - 31 languages, better for translation
  case mltNano = "Fun-ASR-MLT-Nano"

  /// Whether this model type supports translation well
  public var supportsTranslation: Bool {
    self == .mltNano
  }
}

/// Quantization level for Fun-ASR models
public enum FunASRQuantization: String, CaseIterable, Sendable {
  case q4 = "4bit"
  case q8 = "8bit"
  case fp16

  public var isQuantized: Bool {
    self != .fp16
  }
}

/// Combined model variant specification
public struct FunASRModelVariant: Sendable, Equatable {
  public let modelType: FunASRModelType
  public let quantization: FunASRQuantization

  public init(modelType: FunASRModelType = .nano, quantization: FunASRQuantization = .q4) {
    self.modelType = modelType
    self.quantization = quantization
  }

  /// HuggingFace repository ID
  public var repoId: String {
    "mlx-community/\(modelType.rawValue)-2512-\(quantization.rawValue)"
  }

  public var isQuantized: Bool {
    quantization.isQuantized
  }

  public var isMultilingual: Bool {
    modelType == .mltNano
  }

  // Convenience static variants
  public static let nano4bit = FunASRModelVariant(modelType: .nano, quantization: .q4)
  public static let nano8bit = FunASRModelVariant(modelType: .nano, quantization: .q8)
  public static let nanoFP16 = FunASRModelVariant(modelType: .nano, quantization: .fp16)
  public static let mltNano4bit = FunASRModelVariant(modelType: .mltNano, quantization: .q4)
  public static let mltNano8bit = FunASRModelVariant(modelType: .mltNano, quantization: .q8)
  public static let mltNanoFP16 = FunASRModelVariant(modelType: .mltNano, quantization: .fp16)
}

// MARK: - Encoder Configuration

/// Configuration for the SenseVoice encoder
public struct SenseVoiceEncoderConfig: Codable, Sendable {
  /// Input dimension (n_mels * lfr_m = 80 * 7 = 560)
  public var inputDim: Int

  /// Encoder hidden dimension
  public var encoderDim: Int

  /// Number of attention heads
  public var numHeads: Int

  /// Feed-forward network dimension
  public var ffnDim: Int

  /// FSMN kernel size
  public var kernelSize: Int

  /// SANM shift for asymmetric context
  public var sanmShift: Int

  /// Number of initial encoder layers (560 -> 512)
  public var numEncoders0: Int

  /// Number of main encoder layers
  public var numEncoders: Int

  /// Number of time-pooling encoder layers
  public var numTPEncoders: Int

  /// Dropout rate
  public var dropout: Float

  public init(
    inputDim: Int = 560,
    encoderDim: Int = 512,
    numHeads: Int = 4,
    ffnDim: Int = 2048,
    kernelSize: Int = 11,
    sanmShift: Int = 0,
    numEncoders0: Int = 1,
    numEncoders: Int = 49,
    numTPEncoders: Int = 20,
    dropout: Float = 0.0
  ) {
    self.inputDim = inputDim
    self.encoderDim = encoderDim
    self.numHeads = numHeads
    self.ffnDim = ffnDim
    self.kernelSize = kernelSize
    self.sanmShift = sanmShift
    self.numEncoders0 = numEncoders0
    self.numEncoders = numEncoders
    self.numTPEncoders = numTPEncoders
    self.dropout = dropout
  }

  enum CodingKeys: String, CodingKey {
    case inputDim = "input_dim"
    case encoderDim = "encoder_dim"
    case numHeads = "num_heads"
    case ffnDim = "ffn_dim"
    case kernelSize = "kernel_size"
    case sanmShift = "sanm_shift"
    case numEncoders0 = "num_encoders0"
    case numEncoders = "num_encoders"
    case numTPEncoders = "num_tp_encoders"
    case dropout
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    inputDim = try container.decodeIfPresent(Int.self, forKey: .inputDim) ?? 560
    encoderDim = try container.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 512
    numHeads = try container.decodeIfPresent(Int.self, forKey: .numHeads) ?? 4
    ffnDim = try container.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 2048
    kernelSize = try container.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 11
    sanmShift = try container.decodeIfPresent(Int.self, forKey: .sanmShift) ?? 0
    numEncoders0 = try container.decodeIfPresent(Int.self, forKey: .numEncoders0) ?? 1
    numEncoders = try container.decodeIfPresent(Int.self, forKey: .numEncoders) ?? 49
    numTPEncoders = try container.decodeIfPresent(Int.self, forKey: .numTPEncoders) ?? 20
    dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.0
  }
}

// MARK: - Adaptor Configuration

/// Configuration for the audio adaptor
public struct AudioAdaptorConfig: Codable, Sendable {
  /// Downsample rate (group this many frames)
  public var downsampleRate: Int

  /// Input dimension from encoder
  public var encoderDim: Int

  /// Output dimension for LLM
  public var llmDim: Int

  /// Intermediate FFN dimension
  public var ffnDim: Int

  /// Number of transformer blocks
  public var nLayer: Int

  /// Number of attention heads
  public var attentionHeads: Int

  /// Dropout rate
  public var dropout: Float

  public init(
    downsampleRate: Int = 2,
    encoderDim: Int = 512,
    llmDim: Int = 1024,
    ffnDim: Int = 2048,
    nLayer: Int = 2,
    attentionHeads: Int = 8,
    dropout: Float = 0.0
  ) {
    self.downsampleRate = downsampleRate
    self.encoderDim = encoderDim
    self.llmDim = llmDim
    self.ffnDim = ffnDim
    self.nLayer = nLayer
    self.attentionHeads = attentionHeads
    self.dropout = dropout
  }

  enum CodingKeys: String, CodingKey {
    case downsampleRate = "downsample_rate"
    case encoderDim = "encoder_dim"
    case llmDim = "llm_dim"
    case ffnDim = "ffn_dim"
    case nLayer = "n_layer"
    case attentionHeads = "attention_heads"
    case dropout
  }
}

// MARK: - Qwen3 Configuration

/// Configuration for the Qwen3 LLM decoder
public struct Qwen3Config: Codable, Sendable {
  /// Vocabulary size
  public var vocabSize: Int

  /// Hidden dimension
  public var hiddenSize: Int

  /// Number of transformer layers
  public var numHiddenLayers: Int

  /// Number of attention heads
  public var numAttentionHeads: Int

  /// Number of key-value heads (for GQA)
  public var numKeyValueHeads: Int

  /// Intermediate size for MLP
  public var intermediateSize: Int

  /// Maximum position embeddings
  public var maxPositionEmbeddings: Int

  /// RoPE base frequency
  public var ropeTheta: Float

  /// RMS normalization epsilon
  public var rmsNormEps: Float

  /// Whether to tie word embeddings
  public var tieWordEmbeddings: Bool

  /// Attention head dimension
  public var headDim: Int

  public init(
    vocabSize: Int = 151_936,
    hiddenSize: Int = 1024,
    numHiddenLayers: Int = 28,
    numAttentionHeads: Int = 16,
    numKeyValueHeads: Int = 8,
    intermediateSize: Int = 3072,
    maxPositionEmbeddings: Int = 40960,
    ropeTheta: Float = 1_000_000.0,
    rmsNormEps: Float = 1e-6,
    tieWordEmbeddings: Bool = true,
    headDim: Int = 64
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.ropeTheta = ropeTheta
    self.rmsNormEps = rmsNormEps
    self.tieWordEmbeddings = tieWordEmbeddings
    self.headDim = headDim
  }

  enum CodingKeys: String, CodingKey {
    case vocabSize = "vocab_size"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case rmsNormEps = "rms_norm_eps"
    case tieWordEmbeddings = "tie_word_embeddings"
    case headDim = "head_dim"
  }
}

// MARK: - Main Configuration

/// Configuration for the complete Fun-ASR model
public struct FunASRConfig: Codable, Sendable {
  // MARK: - Audio Processing

  /// Sample rate in Hz
  public var sampleRate: Int

  /// Number of mel filterbank bins
  public var nMels: Int

  /// LFR frame stacking count
  public var lfrM: Int

  /// LFR subsampling factor
  public var lfrN: Int

  // MARK: - Component Configs

  /// Encoder configuration
  public var encoder: SenseVoiceEncoderConfig

  /// Adaptor configuration
  public var adaptor: AudioAdaptorConfig

  /// LLM configuration
  public var llm: Qwen3Config

  // MARK: - Special Tokens

  /// Start of speech token
  public var sosToken: String

  /// End of speech token
  public var eosToken: String

  /// Chat template start token
  public var imStartToken: String

  /// Chat template end token
  public var imEndToken: String

  // MARK: - Generation Defaults

  /// Maximum tokens to generate
  public var maxTokens: Int

  /// Default sampling temperature
  public var temperature: Float

  public init(
    sampleRate: Int = 16000,
    nMels: Int = 80,
    lfrM: Int = 7,
    lfrN: Int = 6,
    encoder: SenseVoiceEncoderConfig = SenseVoiceEncoderConfig(),
    adaptor: AudioAdaptorConfig = AudioAdaptorConfig(),
    llm: Qwen3Config = Qwen3Config(),
    sosToken: String = "<|startofspeech|>",
    eosToken: String = "<|endofspeech|>",
    imStartToken: String = "<|im_start|>",
    imEndToken: String = "<|im_end|>",
    maxTokens: Int = 512,
    temperature: Float = 0.0
  ) {
    self.sampleRate = sampleRate
    self.nMels = nMels
    self.lfrM = lfrM
    self.lfrN = lfrN
    self.encoder = encoder
    self.adaptor = adaptor
    self.llm = llm
    self.sosToken = sosToken
    self.eosToken = eosToken
    self.imStartToken = imStartToken
    self.imEndToken = imEndToken
    self.maxTokens = maxTokens
    self.temperature = temperature
  }

  enum CodingKeys: String, CodingKey {
    case sampleRate = "sample_rate"
    case nMels = "n_mels"
    case lfrM = "lfr_m"
    case lfrN = "lfr_n"
    case encoder
    case adaptor
    case llm
    case sosToken = "sos_token"
    case eosToken = "eos_token"
    case imStartToken = "im_start_token"
    case imEndToken = "im_end_token"
    case maxTokens = "max_tokens"
    case temperature
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    // Required fields from config.json
    sampleRate = try container.decode(Int.self, forKey: .sampleRate)
    nMels = try container.decode(Int.self, forKey: .nMels)
    lfrM = try container.decode(Int.self, forKey: .lfrM)
    lfrN = try container.decode(Int.self, forKey: .lfrN)
    encoder = try container.decode(SenseVoiceEncoderConfig.self, forKey: .encoder)
    adaptor = try container.decode(AudioAdaptorConfig.self, forKey: .adaptor)
    llm = try container.decode(Qwen3Config.self, forKey: .llm)
    // Optional fields with defaults (not in model's config.json)
    sosToken = try container.decodeIfPresent(String.self, forKey: .sosToken) ?? "<|startofspeech|>"
    eosToken = try container.decodeIfPresent(String.self, forKey: .eosToken) ?? "<|endofspeech|>"
    imStartToken = try container.decodeIfPresent(String.self, forKey: .imStartToken) ?? "<|im_start|>"
    imEndToken = try container.decodeIfPresent(String.self, forKey: .imEndToken) ?? "<|im_end|>"
    maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 512
    temperature = try container.decodeIfPresent(Float.self, forKey: .temperature) ?? 0.0
  }

  /// Load configuration from a JSON file
  public static func load(from url: URL) throws -> FunASRConfig {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(FunASRConfig.self, from: data)
  }
}

// MARK: - Supported Languages

/// Supported languages for Fun-ASR transcription and translation
public enum FunASRLanguage: String, CaseIterable, Sendable {
  case english = "en"
  case chinese = "zh"
  case japanese = "ja"
  case korean = "ko"
  case spanish = "es"
  case french = "fr"
  case german = "de"
  case italian = "it"
  case portuguese = "pt"
  case russian = "ru"
  case arabic = "ar"
  case thai = "th"
  case vietnamese = "vi"
  case auto

  public var displayName: String {
    switch self {
      case .english: "English"
      case .chinese: "Chinese"
      case .japanese: "Japanese"
      case .korean: "Korean"
      case .spanish: "Spanish"
      case .french: "French"
      case .german: "German"
      case .italian: "Italian"
      case .portuguese: "Portuguese"
      case .russian: "Russian"
      case .arabic: "Arabic"
      case .thai: "Thai"
      case .vietnamese: "Vietnamese"
      case .auto: "Auto-detect"
    }
  }
}

// MARK: - Task Types

/// Task types for Fun-ASR
public enum FunASRTask: String, Sendable {
  case transcribe
  case translate
}
