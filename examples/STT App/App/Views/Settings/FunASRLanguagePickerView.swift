// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting Fun-ASR source language
struct FunASRLanguagePickerView: View {
  @Binding var selectedLanguage: FunASRLanguage
  let isDisabled: Bool

  var body: some View {
    Picker("Language", selection: $selectedLanguage) {
      Text("Auto-detect").tag(FunASRLanguage.auto)

      Section("Common") {
        ForEach(commonLanguages, id: \.self) { language in
          Text(language.displayName).tag(language)
        }
      }

      Section("All Languages") {
        ForEach(otherLanguages, id: \.self) { language in
          Text(language.displayName).tag(language)
        }
      }
    }
    .disabled(isDisabled)
  }

  private var commonLanguages: [FunASRLanguage] {
    [.english, .chinese, .japanese, .korean, .spanish, .french, .german]
  }

  private var otherLanguages: [FunASRLanguage] {
    FunASRLanguage.allCases.filter { language in
      language != .auto && !commonLanguages.contains(language)
    }
  }
}
