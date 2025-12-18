// Copyright © Anthony DePasquale

import Foundation

/// Utilities for splitting text at natural boundaries for TTS processing.
public enum TextSplitter {
  /// Priority order for finding punctuation split points (best to worst)
  private static let punctuationPriority: [Character] = [".", "!", "?", ";", ":", ",", " "]

  /// Split text at the best punctuation boundary near the middle.
  ///
  /// Searches for punctuation in priority order (., !, ?, ;, :, ,, space) starting from
  /// the middle of the text and expanding outward. Returns two halves of the text.
  ///
  /// - Parameters:
  ///   - text: The text to split
  ///   - minLength: Minimum text length to attempt splitting (default: 10)
  /// - Returns: A tuple of (firstHalf, secondHalf), or nil if no split point found
  public static func splitAtPunctuationBoundary(
    _ text: String,
    minLength: Int = 10,
  ) -> (String, String)? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count > minLength else { return nil }

    let middleIndex = trimmed.index(trimmed.startIndex, offsetBy: trimmed.count / 2)

    // Search for punctuation in priority order
    for punct in punctuationPriority {
      // Search expanding from middle (prefer balanced splits)
      // rightOffset starts at 0 to check the middle position first,
      // leftOffset starts at 1 so we don't check the middle twice
      var leftOffset = 1
      var rightOffset = 0
      let maxSearchDistance = trimmed.count / 2

      while leftOffset < maxSearchDistance || rightOffset < maxSearchDistance {
        // Check right of middle first (slightly prefer keeping first half intact)
        if rightOffset < maxSearchDistance {
          let rightIndex = trimmed.index(middleIndex, offsetBy: rightOffset, limitedBy: trimmed.endIndex)
          if let idx = rightIndex, idx < trimmed.endIndex, trimmed[idx] == punct {
            let splitPoint = trimmed.index(after: idx)
            let firstHalf = String(trimmed[..<splitPoint]).trimmingCharacters(in: .whitespaces)
            let secondHalf = String(trimmed[splitPoint...]).trimmingCharacters(in: .whitespaces)
            if !firstHalf.isEmpty, !secondHalf.isEmpty {
              return (firstHalf, secondHalf)
            }
          }
          rightOffset += 1
        }

        // Check left of middle
        if leftOffset < maxSearchDistance {
          let leftIndex = trimmed.index(middleIndex, offsetBy: -leftOffset, limitedBy: trimmed.startIndex)
          if let idx = leftIndex, idx > trimmed.startIndex, trimmed[idx] == punct {
            let splitPoint = trimmed.index(after: idx)
            let firstHalf = String(trimmed[..<splitPoint]).trimmingCharacters(in: .whitespaces)
            let secondHalf = String(trimmed[splitPoint...]).trimmingCharacters(in: .whitespaces)
            if !firstHalf.isEmpty, !secondHalf.isEmpty {
              return (firstHalf, secondHalf)
            }
          }
          leftOffset += 1
        }
      }
    }

    return nil
  }

  /// Recursively split text until all chunks are under the specified character limit.
  ///
  /// - Parameters:
  ///   - text: The text to split
  ///   - maxCharacters: Maximum characters per chunk
  ///   - minSplitLength: Minimum text length to attempt splitting (default: 10)
  /// - Returns: Array of text chunks, each under maxCharacters (if possible)
  public static func splitToMaxLength(
    _ text: String,
    maxCharacters: Int,
    minSplitLength: Int = 10,
  ) -> [String] {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count > maxCharacters else {
      return trimmed.isEmpty ? [] : [trimmed]
    }

    if let (firstHalf, secondHalf) = splitAtPunctuationBoundary(trimmed, minLength: minSplitLength) {
      return splitToMaxLength(firstHalf, maxCharacters: maxCharacters, minSplitLength: minSplitLength)
        + splitToMaxLength(secondHalf, maxCharacters: maxCharacters, minSplitLength: minSplitLength)
    }

    // Could not split - return as-is
    return [trimmed]
  }
}
