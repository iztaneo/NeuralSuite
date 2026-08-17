// Copyright 2026 NeuralSuite Authors.
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

  // 📍 Generación de imágenes y etiquetas sintéticas mediante SynthTextGenerator
  Tensor images, labels;
  SynthTextGenerator::GenerateBatch(images, labels, 4);

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

  std::cout << "\n🎯 Predicción de Caracteres usando CRNNModel::DecodeWord de Biblioteca:\n" << std::flush;
  Tensor final_logits = ocr_model.Forward(images);
  std::string decoded_text = ocr_model.DecodeWord(final_logits, vocab);

  std::cout << "   - Secuencia decodificada: '" << decoded_text << "'\n" << std::flush;
  std::cout << "   - Etiquetas esperadas:    '";
  for (int i = 0; i < 4; ++i) std::cout << vocab[static_cast<int>(labels[i])];
  std::cout << "'\n" << std::flush;

  std::cout << "============================================================\n" << std::flush;
  std::cout << "Demostración de OCR en modo biblioteca finalizada.\n" << std::flush;
  std::cout << "Nota: entrena sobre imágenes sintéticas de ruido 8x8 con 4\n" << std::flush;
  std::cout << "etiquetas fijas; no es reconocimiento de texto real.\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
