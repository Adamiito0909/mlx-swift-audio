// Copyright © Anthony DePasquale

import Foundation
import MLX

@testable import MLXAudio

// MARK: - Shared Test Resources

/// Global shared model for all Chatterbox tests.
///
/// **IMPORTANT: Must Use xcodebuild (NOT swift test)**
///
/// Tests that use MLX/Metal must be run with `xcodebuild`, not `swift test`.
/// Using `swift test` will fail with "Failed to load the default metallib" error.
///
/// **IMPORTANT: Memory Management**
///
/// The Chatterbox model requires ~2GB of GPU memory. When running tests:
///
/// 1. **Run only ONE test suite at a time** to avoid loading multiple model instances
/// 2. Use `xcodebuild test -only-testing:` to run specific tests
/// 3. If memory usage exceeds ~3GB, something is loading multiple models
///
/// Example commands:
/// ```bash
/// # Run only the pipeline benchmark
/// xcodebuild test -scheme mlx-audio-Package -destination 'platform=macOS' \
///   -only-testing:mlx-audioPackageTests/ChatterboxBenchmark/pipelineBenchmark
///
/// # Run only ChatterboxTests
/// xcodebuild test -scheme mlx-audio-Package -destination 'platform=macOS' \
///   -only-testing:mlx-audioPackageTests/ChatterboxTests
///
/// # WRONG - do not use swift test for MLX tests
/// # swift test --filter ChatterboxBenchmark  # Will fail with metallib error
/// ```
///
/// **Do NOT run all tests at once** as this will load multiple models and exhaust memory.
@MainActor
enum ChatterboxTestHelper {
  /// Shared model instance - loaded once and reused across ALL test suites
  private static var _sharedModel: ChatterboxModel?

  /// Get or load the shared model (loads only once)
  ///
  /// Note: Do NOT mix this with ChatterboxEngine in the same test run,
  /// as ChatterboxEngine loads its own copy of the model internally.
  static func getOrLoadModel() async throws -> ChatterboxModel {
    if let model = _sharedModel {
      return model
    }
    print("[ChatterboxTestHelper] Loading shared model (first time)...")
    let model = try await ChatterboxModel.load()
    eval(model)
    _sharedModel = model
    print("[ChatterboxTestHelper] Shared model loaded and cached")
    return model
  }

  /// Clear cached resources (call after tests if needed)
  static func clearCache() {
    _sharedModel = nil
    print("[ChatterboxTestHelper] Cache cleared")
  }
}
