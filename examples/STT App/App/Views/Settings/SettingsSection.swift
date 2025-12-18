// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Combined settings view for STT configuration
struct SettingsSection: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var appState = appState

    Form {
      // Provider Selection
      Section {
        ProviderPickerView(
          selectedProvider: $appState.selectedProvider,
          isDisabled: appState.isTranscribing
        )
      } header: {
        Text("Provider")
      }

      // Provider-specific Model Settings
      Section {
        switch appState.selectedProvider {
          case .whisper:
            whisperModelSettings

          case .funASR:
            funASRModelSettings
        }
      } header: {
        Text("Model")
      }

      // Task Selection
      Section {
        TaskPickerView(
          selectedTask: $appState.selectedTask,
          provider: appState.selectedProvider,
          isEnglishOnlyModel: appState.selectedProvider == .whisper
            && appState.selectedWhisperModelSize.isEnglishOnly,
          isDisabled: appState.isTranscribing
        )

        Text(taskDescription)
          .font(.caption)
          .foregroundStyle(.secondary)
      } header: {
        Text("Task")
      }

      // Language Settings
      Section {
        switch appState.selectedProvider {
          case .whisper:
            whisperLanguageSettings

          case .funASR:
            funASRLanguageSettings
        }
      } header: {
        Text("Language")
      }

      // Whisper-only: Timestamp Settings
      if appState.selectedProvider == .whisper, appState.selectedTask != .detectLanguage {
        Section {
          TimestampPickerView(
            selectedTimestamps: $appState.timestampGranularity,
            isDisabled: appState.isTranscribing
          )

          Text(appState.timestampGranularity.description)
            .font(.caption)
            .foregroundStyle(.secondary)
        } header: {
          Text("Output")
        }
      }
    }
    .formStyle(.grouped)
  }

  // MARK: - Whisper Settings

  @ViewBuilder
  private var whisperModelSettings: some View {
    @Bindable var appState = appState

    ModelPickerView(
      selectedModel: Binding(
        get: { appState.selectedWhisperModelSize },
        set: { appState.setWhisperModelSize($0) }
      ),
      isDisabled: appState.isTranscribing
    )

    QuantizationPickerView(
      selectedQuantization: Binding(
        get: { appState.selectedWhisperQuantization },
        set: { appState.setWhisperQuantization($0) }
      ),
      isDisabled: appState.isTranscribing
    )
  }

  @ViewBuilder
  private var whisperLanguageSettings: some View {
    @Bindable var appState = appState

    LanguagePickerView(
      selectedLanguage: $appState.selectedWhisperLanguage,
      isDisabled: appState.isTranscribing || appState.selectedTask == .detectLanguage
    )
  }

  // MARK: - Fun-ASR Settings

  @ViewBuilder
  private var funASRModelSettings: some View {
    @Bindable var appState = appState

    FunASRModelPickerView(
      selectedModelType: Binding(
        get: { appState.selectedFunASRModelType },
        set: { appState.setFunASRModelType($0) }
      ),
      isDisabled: appState.isTranscribing
    )

    FunASRQuantizationPickerView(
      selectedQuantization: Binding(
        get: { appState.selectedFunASRQuantization },
        set: { appState.setFunASRQuantization($0) }
      ),
      isDisabled: appState.isTranscribing
    )

    // Show MLT hint for translation
    if appState.selectedTask == .translate, appState.selectedFunASRModelType != .mltNano {
      Text("MLT Nano is recommended for translation tasks")
        .font(.caption)
        .foregroundStyle(.orange)
    }
  }

  @ViewBuilder
  private var funASRLanguageSettings: some View {
    @Bindable var appState = appState

    FunASRLanguagePickerView(
      selectedLanguage: $appState.selectedFunASRLanguage,
      isDisabled: appState.isTranscribing
    )

    // Show target language picker for translation
    if appState.selectedTask == .translate {
      TargetLanguagePickerView(
        selectedLanguage: $appState.selectedFunASRTargetLanguage,
        isDisabled: appState.isTranscribing
      )
    }
  }

  // MARK: - Task Description

  private var taskDescription: String {
    switch appState.selectedTask {
      case .transcribe:
        "Transcribe speech in the original language"
      case .translate:
        if appState.selectedProvider == .funASR {
          "Translate speech to \(appState.selectedFunASRTargetLanguage.displayName)"
        } else {
          "Translate speech to English"
        }
      case .detectLanguage:
        "Detect the spoken language"
    }
  }
}
