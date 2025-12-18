// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting STT task type
struct TaskPickerView: View {
  @Binding var selectedTask: STTTask
  let provider: STTProvider
  let isEnglishOnlyModel: Bool
  let isDisabled: Bool

  var body: some View {
    Picker("Task", selection: $selectedTask) {
      ForEach(availableTasks, id: \.self) { task in
        Text(task.rawValue).tag(task)
      }
    }
    .pickerStyle(.menu)
    .disabled(isDisabled)
    .onChange(of: provider) { _, newProvider in
      // Reset to transcribe if current task not available for new provider
      if !selectedTask.isAvailable(for: newProvider) {
        selectedTask = .transcribe
      }
    }
  }

  private var availableTasks: [STTTask] {
    STTTask.allCases.filter { task in
      // Check provider support
      guard task.isAvailable(for: provider) else { return false }

      // For Whisper, English-only models can't translate
      if provider == .whisper, isEnglishOnlyModel, task == .translate {
        return false
      }

      return true
    }
  }
}

extension STTTask {
  var icon: String {
    switch self {
      case .transcribe: "text.alignleft"
      case .translate: "globe"
      case .detectLanguage: "questionmark.circle"
    }
  }

  var description: String {
    switch self {
      case .transcribe:
        "Transcribe speech in the original language"
      case .translate:
        "Translate speech to English"
      case .detectLanguage:
        "Detect the spoken language"
    }
  }
}
