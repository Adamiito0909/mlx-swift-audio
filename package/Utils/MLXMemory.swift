// Copyright © Anthony DePasquale
//
// MLX GPU Memory Configuration Utilities

import Foundation
import MLX

/// Configuration for MLX GPU memory management.
///
/// MLX uses a buffer recycling system that caches intermediate computation buffers.
/// Without limits, this cache can grow to several GB during long inference runs.
/// This utility provides APIs to configure memory limits and monitor usage.
///
/// ## Usage
///
/// Configure memory limits before loading models:
/// ```swift
/// // Set a 512MB cache limit
/// MLXMemory.configure(cacheLimit: 512 * 1024 * 1024)
///
/// // Or use platform-appropriate defaults
/// MLXMemory.configureForPlatform()
///
/// // Load and use model...
///
/// // Clear cache between heavy operations
/// MLXMemory.clearCache()
/// ```
///
/// ## Platform Recommendations
///
/// - **iOS**: Use aggressive limits (256-512MB) due to jetsam memory limits
/// - **macOS**: Can use more relaxed limits (512MB-1GB)
/// - **Default**: No limit (MLX default behavior)
///
/// ## Monitoring
///
/// Use `MLXMemory.snapshot()` to monitor memory usage:
/// ```swift
/// let stats = MLXMemory.snapshot()
/// print("Active: \(stats.activeMB)MB, Cache: \(stats.cacheMB)MB")
/// ```
public enum MLXMemory {
  /// Memory statistics snapshot
  public struct Snapshot {
    /// Memory actively used by MLXArrays (MB)
    public let activeMB: Int
    /// Cached memory available for reuse (MB)
    public let cacheMB: Int
    /// Peak memory usage since start or last reset (MB)
    public let peakMB: Int
    /// Total memory (active + cache) in MB
    public var totalMB: Int { activeMB + cacheMB }
  }

  /// Get current memory statistics
  public static func snapshot() -> Snapshot {
    Snapshot(
      activeMB: Memory.activeMemory / 1024 / 1024,
      cacheMB: Memory.cacheMemory / 1024 / 1024,
      peakMB: Memory.peakMemory / 1024 / 1024,
    )
  }

  /// Configure GPU cache limit.
  ///
  /// Setting a cache limit prevents unbounded memory growth during inference.
  /// The limit controls the buffer pool size, not total GPU memory usage.
  ///
  /// - Parameter cacheLimit: Maximum cache size in bytes. Pass `nil` for no limit.
  public static func configure(cacheLimit: Int?) {
    if let limit = cacheLimit {
      Memory.cacheLimit = limit
      Log.tts.debug("[MLXMemory] Cache limit set to \(limit / 1024 / 1024)MB")
    }
  }

  /// Configure memory limits appropriate for the current platform.
  ///
  /// - iOS: 512MB cache limit (conservative for jetsam)
  /// - macOS: 1GB cache limit (more relaxed)
  public static func configureForPlatform() {
    #if os(iOS)
    // iOS: Conservative limit due to jetsam
    configure(cacheLimit: 512 * 1024 * 1024)
    #elseif os(macOS)
    // macOS: More relaxed limit
    configure(cacheLimit: 1024 * 1024 * 1024)
    #else
    // Other platforms: Use default (no limit)
    Log.tts.debug("[MLXMemory] No cache limit set for this platform")
    #endif
  }

  /// Clear the GPU buffer cache.
  ///
  /// Call this between heavy operations (e.g., between TTS generations)
  /// to release cached buffers and reduce memory footprint.
  public static func clearCache() {
    Memory.clearCache()
    let stats = snapshot()
    Log.tts.debug("[MLXMemory] Cache cleared. Active: \(stats.activeMB)MB, Cache: \(stats.cacheMB)MB")
  }

  /// Reset peak memory tracking.
  ///
  /// Call this before a section you want to measure.
  public static func resetPeakMemory() {
    GPU.resetPeakMemory()
  }

  /// Log current memory statistics.
  public static func logStats(prefix: String = "") {
    let stats = snapshot()
    let message =
      "\(prefix.isEmpty ? "" : prefix + " ")Active: \(stats.activeMB)MB, Cache: \(stats.cacheMB)MB, Peak: \(stats.peakMB)MB"
    Log.tts.info("[MLXMemory] \(message)")
  }
}
