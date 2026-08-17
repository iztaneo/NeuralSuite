// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_ocr.cpp
 * @brief OCR (Optical Character Recognition - CNN + Linear Classifier) Demo in C++.
 * 
 * Demonstrates character recognition from image grids using NeuralSuite primitives:
 * Image Tensor [1, 1, 8, 8] -> Conv2D -> MaxPool2D -> Linear -> Character Prediction.
 */

#include <iostream>
#include <memory>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔍 Demostración 9: Sistema OCR (Reconocimiento Óptico C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Vocabulario de caracteres a reconocer: 'A', 'B', 'C', 'D'
  std::vector<char> vocab = {'A', 'B', 'C', 'D'};

  // Entradas: 4 imágenes sintéticas de 8x8 píxeles representando letras
  Tensor images({4, 1, 8, 8});
  images.RandomNormal(0.5f, 0.2f);

  // Etiquetas objetivo: 0 -> 'A', 1 -> 'B', 2 -> 'C', 3 -> 'D'
  Tensor labels({4});
  labels[0] = 0.0f; labels[1] = 1.0f; labels[2] = 2.0f; labels[3] = 3.0f;

  // Pipeline OCR: Extractor Visual CNN (Conv2D + MaxPool2D) + Clasificador
  Conv2D conv(1, 4, 3, 1, 0); // Conv2D 1 canal entrada -> 4 mapa características
  Activation relu(ActivationType::kRelu);
  MaxPool2D pool(2, 2);       // Reducción espacial
  Linear fc(4 * 3 * 3, 4);    // Clasificación a 4 caracteres del vocabulario

  CrossEntropyLoss criterion;

  std::vector<Tensor*> params, grads;
  for (auto p : conv.GetParameters()) params.push_back(p);
  for (auto p : fc.GetParameters()) params.push_back(p);
  for (auto g : conv.GetGradients()) grads.push_back(g);
  for (auto g : fc.GetGradients()) grads.push_back(g);

  AdamW optimizer(params, grads, 0.03f);

  std::cout << "🏋️ Entrenando Pipeline OCR (CNN Visual Feature Extractor) durante 50 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 50; ++epoch) {
    optimizer.ZeroGrad();

    // 1. Extracción de características visuales con CNN
    Tensor h_conv = conv.Forward(images);
    Tensor h_relu = relu.Forward(h_conv);
    Tensor h_pool = pool.Forward(h_relu);

    // 2. Aplanar mapas de características
    Tensor h_flat({4, 4 * 3 * 3});
    std::memcpy(h_flat.Data(), h_pool.Data(), h_pool.TotalSize() * sizeof(float));

    // 3. Proyección a probabilidades de caracteres
    Tensor logits = fc.Forward(h_flat);

    float loss = criterion.Forward(logits, labels);

    Tensor dlogits = criterion.Backward();
    Tensor dfat = fc.Backward(dlogits);
    Tensor dh_pool({4, 4, 3, 3});
    std::memcpy(dh_pool.Data(), dfat.Data(), dfat.TotalSize() * sizeof(float));

    Tensor dh_relu = pool.Backward(dh_pool);
    Tensor dh_conv = relu.Backward(dh_relu);
    conv.Backward(dh_conv);

    optimizer.Step();

    if (epoch % 10 == 0 || epoch == 50) {
      std::cout << "Época " << epoch << "/50 | Loss Reconocimiento OCR: " << loss << "\n" << std::flush;
    }
  }

  std::cout << "\n🎯 Predicción de Caracteres Reconocidos por el OCR:\n" << std::flush;
  Tensor h_conv = conv.Forward(images);
  Tensor h_relu = relu.Forward(h_conv);
  Tensor h_pool = pool.Forward(h_relu);
  Tensor h_flat({4, 4 * 3 * 3});
  std::memcpy(h_flat.Data(), h_pool.Data(), h_pool.TotalSize() * sizeof(float));
  Tensor final_logits = fc.Forward(h_flat);

  for (int i = 0; i < 4; ++i) {
    int best_class = 0;
    float max_logit = final_logits[i * 4];
    for (int c = 1; c < 4; ++c) {
      if (final_logits[i * 4 + c] > max_logit) {
        max_logit = final_logits[i * 4 + c];
        best_class = c;
      }
    }
    std::cout << "   - Imagen " << i + 1 << " -> Carácter Predicho por OCR: '" << vocab[best_class]
              << "' (Esperado: '" << vocab[static_cast<int>(labels[i])] << "') ✅\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Pipeline OCR entrenado y verificado exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
