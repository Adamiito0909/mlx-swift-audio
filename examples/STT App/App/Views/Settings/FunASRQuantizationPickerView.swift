// Copyright © Anthony DePasquale

import MLXAudio
import SwiftUI

/// Picker for selecting Fun-ASR quantization level
struct FunASRQuantizationPickerView: View {
  @Binding var selectedQuantization: FunASRQuantization
  let isDisabled: Bool

  var body: some View {
    Picker("Quantization", selection: $selectedQuantization) {
      ForEach(FunASRQuantization.allCases, id: \.self) { quantization in
        Text(quantization.displayName).tag(quantization)
      }
    }
    .disabled(isDisabled)
  }
}
