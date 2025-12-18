// Copyright © Anthony DePasquale

import MLX

// MARK: - Reverse Along Axis

/// Reverse array along a specific axis
///
/// - Parameters:
///   - x: Input array
///   - axis: Axis to reverse along (negative values count from end)
/// - Returns: Array with elements reversed along the specified axis
func reverseAlongAxis(_ x: MLXArray, axis: Int) -> MLXArray {
  let actualAxis = axis < 0 ? x.ndim + axis : axis
  let size = x.shape[actualAxis]
  // Create reversed indices using MLX operations (avoid Swift array allocation)
  // indices = [size-1, size-2, ..., 1, 0]
  let indices = MLXArray(Int32(size - 1)) - MLXArray(0 ..< size).asType(.int32)
  return MLX.take(x, indices, axis: actualAxis)
}

// MARK: - arange extension

// TODO: Remove after next mlx-swift release. `arange` has been added to MLX Swift:
// https://github.com/ml-explore/mlx-swift/pull/302

extension MLXArray {
  /// Generate values in the half-open interval `[0, stop)`.
  ///
  /// Example:
  /// ```swift
  /// let r = MLXArray.arange(10) // [0, 1, 2, ..., 9]
  /// ```
  ///
  /// - Parameter stop: End of the sequence (exclusive).
  /// - Returns: An array containing the generated range as int32.
  static func arange(_ stop: Int) -> MLXArray {
    guard stop > 0 else { return MLXArray([Int32]()) }
    return MLXArray(Array(0 ..< stop).map { Int32($0) })
  }
}
