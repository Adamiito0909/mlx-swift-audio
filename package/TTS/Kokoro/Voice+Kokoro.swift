// Copyright © Hexgrad (original model implementation)
// Ported to MLX from https://github.com/hexgrad/kokoro
// Copyright © 2024 Prince Canuma and contributors to Blaizzy/mlx-audio
// Copyright © Anthony DePasquale
// License: licenses/kokoro.txt

import Foundation
import MLXAudio

extension Voice {
  /// Mapping from single-letter voice prefixes to country codes
  /// Used for Kokoro-style voice names (e.g., "af_heart" -> American Female)
  private static let prefixToCountry: [Character: String] = [
    "a": "US", // American
    "b": "GB", // British
    "e": "ES", // Spanish
    "f": "FR", // French
    "h": "IN", // Hindi
    "i": "IT", // Italian
    "j": "JP", // Japanese
    "p": "BR", // Portuguese (Brazil)
    "z": "CN", // Chinese
  ]

  /// Create a Voice from a Kokoro-style voice identifier (e.g., "af_heart")
  public static func fromKokoroID(_ id: String) -> Voice {
    let displayName = formatKokoroDisplayName(id)
    let languageCode = inferLanguageCode(from: id)
    return Voice(id: id, displayName: displayName, languageCode: languageCode)
  }

  private static func formatKokoroDisplayName(_ id: String) -> String {
    guard id.count >= 3 else { return id.capitalized }
    // Format: "af_heart" -> "Heart"
    let name = id.dropFirst(3)
    return name.capitalized
  }

  private static func inferLanguageCode(from id: String) -> String {
    guard let firstChar = id.first else { return "en-US" }
    let country = prefixToCountry[firstChar] ?? "US"
    let language =
      country == "US" || country == "GB" ? "en" :
      country == "ES" ? "es" :
      country == "FR" ? "fr" :
      country == "IT" ? "it" :
      country == "JP" ? "ja" :
      country == "CN" ? "zh" :
      country == "BR" ? "pt" :
      country == "IN" ? "hi" : "en"
    return "\(language)-\(country)"
  }
}
