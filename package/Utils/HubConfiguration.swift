// Copyright © Anthony DePasquale

import Foundation
import Hub

/// Shared Hub API configuration for model downloads
///
/// Uses the system caches directory instead of Documents:
/// - macOS: ~/Library/Caches/huggingface/
/// - iOS: <app-sandbox>/Library/Caches/huggingface/
///
/// This is appropriate for ML models because:
/// - Models can be re-downloaded if purged
/// - Not backed up to iCloud (saves user storage)
/// - System can reclaim space on iOS if needed
public enum HubConfiguration {
  /// Shared HubApi instance configured to use the caches directory
  public static let shared: HubApi = {
    let cachesURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    let downloadBase = cachesURL.appending(component: "huggingface")
    return HubApi(downloadBase: downloadBase)
  }()

  /// The base directory where models are cached
  public static var cacheDirectory: URL {
    let cachesURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
    return cachesURL.appending(component: "huggingface")
  }

  /// Clear all cached models
  public static func clearCache() throws {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: cacheDirectory.path) {
      try fileManager.removeItem(at: cacheDirectory)
    }
  }
}
