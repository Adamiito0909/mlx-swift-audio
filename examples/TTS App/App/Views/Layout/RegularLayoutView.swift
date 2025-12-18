// Copyright © Anthony DePasquale

import SwiftUI

struct RegularLayoutView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState
    NavigationSplitView {
      ScrollView {
        SettingsSection()
          .padding()
      }
      .navigationTitle("Settings")
      .navigationSplitViewColumnWidth(min: 250, ideal: 280, max: 350)
    } detail: {
      VStack(spacing: 0) {
        ScrollView {
          VStack(spacing: 24) {
            InputSection()
            OutputSection()
          }
          .padding()
        }
      }
      .navigationTitle("TTS App")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
        .toolbar {
          ToolbarItemGroup {
            VoicePickerView()
            ProviderPickerView(
              selectedProvider: appState.selectedProvider,
              onSelect: { provider in
                Task { await appState.selectProvider(provider) }
              },
            )
          }
        }
    }
  }
}
