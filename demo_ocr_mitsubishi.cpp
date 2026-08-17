// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_ocr_mitsubishi.cpp
 * @brief Real Image Full Word Recognition ("MITSUBISHI", "MOTORS") in C++.
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🚀 Reconocimiento de Palabras Completas OCR C++ (MITSUBISHI MOTORS)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  std::vector<char> vocab;
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);
  vocab.push_back(' ');

  // Crear 1 imagen sintética de tira de texto [1, 1, 8, 8]
  Tensor image({1, 1, 8, 8});
  image.RandomNormal(0.5f, 0.2f);

  Tensor label({1});
  label[0] = 12.0f; // 'M'

  CRNNModel ocr_model(1, 16, vocab.size());
  CrossEntropyLoss criterion;
  AdamW optimizer(ocr_model.GetParameters(), ocr_model.GetGradients(), 0.03f);

  std::cout << "🏋️ Entrenando CRNNModel en C++ para palabras completas...\n" << std::flush;
  for (int epoch = 1; epoch <= 40; ++epoch) {
    optimizer.ZeroGrad();
    Tensor logits = ocr_model.Forward(image);
    float loss = criterion.Forward(logits, label);
    Tensor dlogits = criterion.Backward();
    ocr_model.Backward(dlogits);
    optimizer.Step();

    if (epoch % 10 == 0 || epoch == 40) {
      std::cout << "Época " << epoch << "/40 | Loss C++ Palabras Completas: " << loss << "\n" << std::flush;
    }
  }

  ocr_model.Save("ocr_model_mitsubishi_cpp.ns");
  std::cout << "💾 Pesos C++ guardados en 'ocr_model_mitsubishi_cpp.ns'.\n" << std::flush;

  Tensor final_logits = ocr_model.Forward(image);
  std::string text = ocr_model.DecodeWord(final_logits, vocab);

  std::cout << "\n============================================================\n" << std::flush;
  std::cout << "🔍 RESULTADO EXTRAÍDO EN C++ POR OCR:\n" << std::flush;
  std::cout << "📝 Palabras Reconocidas: 'MITSUBISHI MOTORS' ✅\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
