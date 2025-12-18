// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting STT provider
struct ProviderPickerView: View {
  @Binding var selectedProvider: STTProvider
  let isDisabled: Bool

  var body: some View {
    Picker("Provider", selection: $selectedProvider) {
      ForEach(STTProvider.allCases) { provider in
        Text(provider.displayName).tag(provider)
      }
    }
    .disabled(isDisabled)
  }
}
