// Copyright © Anthony DePasquale

import SwiftUI
import UniformTypeIdentifiers

/// Configuration for the reference audio picker
struct ReferenceAudioPickerConfig {
  /// Title displayed at the top
  var title: String = "Reference Audio"

  /// Description text shown at the bottom
  var infoText: String

  /// Loading state text
  var loadingText: String = "Processing..."
}

/// Reusable reference audio picker component for TTS models
struct ReferenceAudioPicker: View {
  let config: ReferenceAudioPickerConfig

  /// Current status description
  let statusDescription: String

  /// Whether audio is loaded
  let isLoaded: Bool

  /// Callbacks
  let onLoadDefault: () async throws -> Void
  let onLoadFromFile: (URL) async throws -> Void
  let onLoadFromURL: (URL) async throws -> Void

  @State private var isShowingFilePicker = false
  @State private var isShowingURLInput = false
  @State private var customURL = ""
  @State private var isLoading = false
  @State private var errorMessage: String?

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text(config.title)
        .font(.headline)

      // Current selection display
      HStack {
        Image(systemName: statusIcon)
          .foregroundStyle(statusColor)

        VStack(alignment: .leading, spacing: 2) {
          Text(statusDescription)
            .font(.subheadline)

          if isLoading {
            Text(config.loadingText)
              .font(.caption)
              .foregroundStyle(.secondary)
          } else if let error = errorMessage {
            Text(error)
              .font(.caption)
              .foregroundStyle(.red)
          }
        }

        Spacer()
      }
      .padding(8)
      .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))

      // Action buttons
      HStack(spacing: 12) {
        Button {
          Task { await loadDefault() }
        } label: {
          Label("Default", systemImage: "waveform")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)

        Button {
          isShowingFilePicker = true
        } label: {
          Label("File", systemImage: "folder")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)

        Button {
          isShowingURLInput = true
        } label: {
          Label("URL", systemImage: "link")
        }
        .buttonStyle(.bordered)
        .disabled(isLoading)
      }

      // Info text
      Text(config.infoText)
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .fileImporter(
      isPresented: $isShowingFilePicker,
      allowedContentTypes: [.audio, .wav, .mp3, .aiff],
      allowsMultipleSelection: false
    ) { result in
      handleFileSelection(result)
    }
    .alert("Enter Audio URL", isPresented: $isShowingURLInput) {
      TextField("https://...", text: $customURL)
        .textContentType(.URL)
      #if os(iOS)
        .autocapitalization(.none)
      #endif

      Button("Cancel", role: .cancel) {
        customURL = ""
      }

      Button("Load") {
        Task { await loadFromURL() }
      }
      .disabled(customURL.isEmpty)
    } message: {
      Text("Enter the URL of an audio file (MP3, WAV, etc.)")
    }
  }

  // MARK: - Computed Properties

  private var statusIcon: String {
    if isLoading {
      "arrow.triangle.2.circlepath"
    } else if isLoaded {
      "checkmark.circle.fill"
    } else {
      "circle"
    }
  }

  private var statusColor: Color {
    if isLoading {
      .orange
    } else if isLoaded {
      .green
    } else {
      .secondary
    }
  }

  // MARK: - Actions

  private func loadDefault() async {
    isLoading = true
    errorMessage = nil

    do {
      try await onLoadDefault()
    } catch {
      errorMessage = error.localizedDescription
    }

    isLoading = false
  }

  private func handleFileSelection(_ result: Result<[URL], Error>) {
    switch result {
      case let .success(urls):
        guard let url = urls.first else { return }

        guard url.startAccessingSecurityScopedResource() else {
          errorMessage = "Permission denied"
          return
        }

        Task {
          defer { url.stopAccessingSecurityScopedResource() }

          isLoading = true
          errorMessage = nil

          do {
            try await onLoadFromFile(url)
          } catch {
            errorMessage = error.localizedDescription
          }

          isLoading = false
        }

      case let .failure(error):
        errorMessage = error.localizedDescription
    }
  }

  private func loadFromURL() async {
    guard let url = URL(string: customURL) else {
      errorMessage = "Invalid URL"
      return
    }

    isLoading = true
    errorMessage = nil

    do {
      try await onLoadFromURL(url)
    } catch {
      errorMessage = error.localizedDescription
    }

    isLoading = false
    customURL = ""
  }
}

// MARK: - UTType Extensions

extension UTType {
  static var wav: UTType { UTType(filenameExtension: "wav")! }
  static var mp3: UTType { UTType(filenameExtension: "mp3")! }
  static var aiff: UTType { UTType(filenameExtension: "aiff")! }
}
