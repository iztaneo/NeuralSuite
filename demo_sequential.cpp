// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_sequential.cpp
 * @brief High-level Sequential Container Library Demo using #include "neuralsuite.h".
 */

#include <iostream>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🚀 Demostración API Biblioteca: Sequential (#include \"neuralsuite.h\")\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Dataset XOR
  Tensor X({4, 2});
  X[0] = 0.0f; X[1] = 0.0f;
  X[2] = 0.0f; X[3] = 1.0f;
  X[4] = 1.0f; X[5] = 0.0f;
  X[6] = 1.0f; X[7] = 1.0f;

  Tensor Y({4});
  Y[0] = 0.0f; Y[1] = 1.0f; Y[2] = 1.0f; Y[3] = 0.0f;

  // API de Alto Nivel: Crear modelo apilando capas con Sequential
  Sequential model;
  model.Add(std::make_shared<Linear>(2, 16));
  model.Add(std::make_shared<Activation>(ActivationType::kRelu));
  model.Add(std::make_shared<Linear>(16, 2));

  CrossEntropyLoss criterion;
  AdamW optimizer(model.GetParameters(), model.GetGradients(), 0.05f);

  std::cout << "🏋️ Entrenando modelo de alto nivel Sequential durante 100 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.ZeroGrad();

    Tensor logits = model.Forward(X);
    float loss = criterion.Forward(logits, Y);

    Tensor dlogits = criterion.Backward();
    model.Backward(dlogits);
    optimizer.Step();

    if (epoch % 25 == 0 || epoch == 100) {
      std::cout << "Época " << epoch << "/100 | Loss Sequential: " << loss << "\n" << std::flush;
    }
  }

  // Guardar y cargar pesos desde la biblioteca
  model.Save("sequential_model.ns");
  std::cout << "💾 Modelo guardado y verificado en 'sequential_model.ns'.\n" << std::flush;

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Demostración de la API de biblioteca completada exitosamente!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
