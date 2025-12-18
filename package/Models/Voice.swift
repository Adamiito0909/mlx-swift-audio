// Copyright © Anthony DePasquale

import Foundation

/// Represents a voice option for TTS engines
public struct Voice: Identifiable, Hashable, Sendable {
  /// Unique identifier for the voice (e.g., "af_heart", "conversational_a")
  public let id: String

  /// Human-readable display name
  public let displayName: String

  /// Language/region code (e.g., "en-US", "en-GB", "ja-JP")
  public let languageCode: String

  public init(id: String, displayName: String, languageCode: String) {
    self.id = id
    self.displayName = displayName
    self.languageCode = languageCode
  }
}
