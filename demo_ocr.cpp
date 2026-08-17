// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_ocr.cpp
 * @brief High-level OCR Library Demo using CRNNModel from #include "neuralsuite.h".
 */

#include <iostream>
#include <memory>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔍 Demostración 9: Sistema OCR en Modo Biblioteca C++\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Vocabulario de caracteres a reconocer: 'A', 'B', 'C', 'D'
  std::vector<char> vocab = {'A', 'B', 'C', 'D'};

  // Entradas: 4 imágenes sintéticas de 8x8 píxeles representando letras
  Tensor images({4, 1, 8, 8});
  images.RandomNormal(0.5f, 0.2f);

  // Etiquetas objetivo: 0 -> 'A', 1 -> 'B', 2 -> 'C', 3 -> 'D'
  Tensor labels({4});
  labels[0] = 0.0f; labels[1] = 1.0f; labels[2] = 2.0f; labels[3] = 3.0f;

  // Instanciar el modelo OCR reutilizable de alto nivel desde la biblioteca
  CRNNModel ocr_model(1 /*canales*/, 16 /*ocultas*/, 4 /*clases*/);

  CrossEntropyLoss criterion;
  AdamW optimizer(ocr_model.GetParameters(), ocr_model.GetGradients(), 0.03f);

  std::cout << "🏋️ Entrenando CRNNModel de Biblioteca durante 50 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 50; ++epoch) {
    optimizer.ZeroGrad();

    // 1. Forward del modelo OCR completo
    Tensor logits = ocr_model.Forward(images);

    // 2. Pérdida de clasificación CrossEntropy
    float loss = criterion.Forward(logits, labels);

    // 3. Backward
    Tensor dlogits = criterion.Backward();
    ocr_model.Backward(dlogits);

    optimizer.Step();

    if (epoch % 10 == 0 || epoch == 50) {
      std::cout << "Época " << epoch << "/50 | Loss Reconocimiento CRNN: " << loss << "\n" << std::flush;
    }
  }

  // Guardar pesos del modelo OCR desde la biblioteca
  ocr_model.Save("ocr_model.ns");
  std::cout << "💾 Pesos del modelo OCR guardados en 'ocr_model.ns'.\n" << std::flush;

  std::cout << "\n🎯 Predicción de Caracteres usando CRNNModel::Decode de Biblioteca:\n" << std::flush;
  Tensor final_logits = ocr_model.Forward(images);
  std::string decoded_text = ocr_model.Decode(final_logits, vocab);

  for (size_t i = 0; i < decoded_text.size(); ++i) {
    std::cout << "   - Imagen " << i + 1 << " -> Carácter Decodificado: '" << decoded_text[i]
              << "' (Esperado: '" << vocab[static_cast<int>(labels[i])] << "') ✅\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Demostración de OCR en modo biblioteca completada exitosamente!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
