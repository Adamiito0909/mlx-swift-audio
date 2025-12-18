// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting Fun-ASR model type
struct FunASRModelPickerView: View {
  @Binding var selectedModelType: FunASRModelType
  let isDisabled: Bool

  var body: some View {
    Picker("Model", selection: $selectedModelType) {
      ForEach(FunASRModelType.allCases, id: \.self) { modelType in
        Text(modelType.displayName).tag(modelType)
      }
    }
    .disabled(isDisabled)
  }
}
