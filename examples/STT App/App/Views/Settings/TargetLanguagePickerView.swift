// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting Fun-ASR target language (for translation)
struct TargetLanguagePickerView: View {
  @Binding var selectedLanguage: FunASRLanguage
  let isDisabled: Bool

  var body: some View {
    Picker("Target Language", selection: $selectedLanguage) {
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
    // All languages except auto (can't translate to "auto") and common ones
    FunASRLanguage.allCases.filter { language in
      language != .auto && !commonLanguages.contains(language)
    }
  }
}
