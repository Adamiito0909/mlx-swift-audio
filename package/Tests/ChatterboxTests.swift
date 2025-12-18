// Copyright © Anthony DePasquale

import Foundation
import MLX
import Testing

@testable import MLXAudio

// IMPORTANT: See ChatterboxTestHelper.swift for memory management notes.
// Run only ONE test suite at a time to avoid loading multiple ~2GB models.
//
// These tests are DISABLED by default to avoid memory issues when running benchmarks.
// To run these tests, remove the .disabled trait below.

@Suite(.serialized, .disabled("Disabled to avoid loading multiple models during benchmarks"))
struct ChatterboxTests {
  @Test @MainActor func chatterboxEngineInitializes() async {
    let engine = ChatterboxEngine()
    #expect(engine.isLoaded == false)
    #expect(engine.isGenerating == false)
    #expect(engine.provider == .chatterbox)
  }

  @Test @MainActor func chatterboxEngineLoadsModel() async throws {
    // Note: Creates own engine - only run this test suite in isolation
    let engine = ChatterboxEngine()
    try await engine.load()
    #expect(engine.isLoaded == true)
    print("Chatterbox model loaded successfully")
  }

  @Test @MainActor func chatterboxGeneratesAudio() async throws {
    // Note: Creates own engine - only run this test suite in isolation
    let engine = ChatterboxEngine()
    try await engine.load()
    #expect(engine.isLoaded == true)

    // Prepare reference audio (using default sample)
    print("Preparing reference audio...")
    let referenceAudio = try await engine.prepareDefaultReferenceAudio()
    print("Reference audio prepared: \(referenceAudio.description)")

    // Generate speech
    let testText = "Hello, this is a test of the Chatterbox text to speech system."
    print("Generating speech for: \"\(testText)\"")

    let result = try await engine.generate(testText, referenceAudio: referenceAudio)

    switch result {
      case let .samples(data, sampleRate, processingTime):
        print("Generated \(data.count) samples at \(sampleRate)Hz in \(String(format: "%.2f", processingTime)) sec.")
        #expect(data.count > 0)
        #expect(sampleRate == ChatterboxS3GenSr)

        let audioDuration = Double(data.count) / Double(sampleRate)
        let rtf = processingTime / audioDuration
        print("Audio duration: \(String(format: "%.2f", audioDuration)) sec., RTF: \(String(format: "%.2f", rtf))")

      case .file:
        Issue.record("Expected samples result, got file")
    }

    #expect(engine.generationTime > 0)
    print("Chatterbox generation test passed!")
  }

  @Test @MainActor func chatterboxMultipleSpeakers() async throws {
    // Note: Creates own engine - only run this test suite in isolation
    let engine = ChatterboxEngine()
    try await engine.load()

    // Prepare default reference audio
    print("Preparing reference audio...")
    let speaker = try await engine.prepareDefaultReferenceAudio()

    // Generate with the same speaker multiple times (should be fast after first)
    print("Generating with speaker...")
    let result1 = try await engine.generate("First sentence.", referenceAudio: speaker)
    let result2 = try await engine.generate("Second sentence.", referenceAudio: speaker)

    // Verify both generated successfully
    if case let .samples(data1, _, _) = result1 {
      #expect(data1.count > 0)
    }
    if case let .samples(data2, _, _) = result2 {
      #expect(data2.count > 0)
    }

    print("Multiple speaker test passed!")
  }

  @Test @MainActor func chatterboxModelDirectLoad() async throws {
    print("Testing direct ChatterboxModel.load()...")

    // Use global shared model to avoid loading multiple models
    let model = try await ChatterboxTestHelper.getOrLoadModel()

    print("Model loaded successfully (shared)")

    // Check model components are initialized
    // (t3, s3gen, ve are non-optional Module properties, so just access them to verify)
    _ = model.t3
    _ = model.s3gen
    _ = model.ve
    #expect(model.textTokenizer != nil)

    print("ChatterboxModel direct load test passed!")
  }
}
