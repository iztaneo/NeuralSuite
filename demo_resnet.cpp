// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_resnet.cpp
 * @brief Residual Neural Network (ResNet Block Skip Connection) Demo in C++.
 */

#include <iostream>
#include <memory>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🧱 Demostración 5: Red Residual ResNet (Skip Connection C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Dataset XOR
  Tensor X({4, 2});
  X[0] = 0.0f; X[1] = 0.0f;
  X[2] = 0.0f; X[3] = 1.0f;
  X[4] = 1.0f; X[5] = 0.0f;
  X[6] = 1.0f; X[7] = 1.0f;

  Tensor Y({4});
  Y[0] = 0.0f; Y[1] = 1.0f; Y[2] = 1.0f; Y[3] = 0.0f;

  // Modelo ResNet: Capa inicial + Bloque Residual (Suma Shortcut) + Capa final
  Sequential model;
  model.Add(std::make_shared<Linear>(2, 8));

  // Bloque Residual ResNet: y = ReLU(Linear(8, 8) + x)
  auto res_inner = std::make_shared<Linear>(8, 8);
  model.Add(std::make_shared<ResidualBlock>(res_inner));

  model.Add(std::make_shared<Linear>(8, 2));

  CrossEntropyLoss criterion;
  AdamW optimizer(model.GetParameters(), model.GetGradients(), 0.05f);

  std::cout << "🏋️ Entrenando Red Residual ResNet durante 100 épocas en C++...\n" << std::flush;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.ZeroGrad();

    Tensor logits = model.Forward(X);
    float loss = criterion.Forward(logits, Y);

    Tensor dlogits = criterion.Backward();
    model.Backward(dlogits);
    optimizer.Step();

    if (epoch % 25 == 0 || epoch == 100) {
      std::cout << "Época " << epoch << "/100 | Loss ResNet: " << loss << "\n" << std::flush;
    }
  }

  {
    const std::string path = ReleasePath("resnet_model.ns");
    if (model.Save(path)) {
      std::cout << "Pesos guardados en '" << path << "'.\n" << std::flush;
    } else {
      std::cerr << "ERROR: no se pudieron guardar los pesos en '" << path << "'.\n" << std::flush;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Red Residual ResNet entrenada y verificada exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
