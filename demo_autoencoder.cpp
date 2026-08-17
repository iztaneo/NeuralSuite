// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_autoencoder.cpp
 * @brief Autoencoder (Encoder-Decoder Bottleneck Reconstruction) Demo in C++.
 */

#include <iostream>
#include <memory>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔄 Demostración 4: Autoencoder (Compresión y Reconstrucción C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Dataset de vectores de 8 dimensiones para comprimir a 2 dimensiones (cuello de botella)
  Tensor X({4, 8});
  X.RandomNormal(0.0f, 1.0f);

  // Codificador (Encoder): 8 entradas -> 4 ocultas -> 2 latentes
  Sequential encoder;
  encoder.Add(std::make_shared<Linear>(8, 4));
  encoder.Add(std::make_shared<Activation>(ActivationType::kRelu));
  encoder.Add(std::make_shared<Linear>(4, 2));

  // Decodificador (Decoder): 2 latentes -> 4 ocultas -> 8 reconstruidas
  Sequential decoder;
  decoder.Add(std::make_shared<Linear>(2, 4));
  decoder.Add(std::make_shared<Activation>(ActivationType::kRelu));
  decoder.Add(std::make_shared<Linear>(4, 8));

  MSELoss criterion;

  std::vector<Tensor*> params, grads;
  for (auto p : encoder.GetParameters()) params.push_back(p);
  for (auto p : decoder.GetParameters()) params.push_back(p);
  for (auto g : encoder.GetGradients()) grads.push_back(g);
  for (auto g : decoder.GetGradients()) grads.push_back(g);

  AdamW optimizer(params, grads, 0.03f);

  std::cout << "🏋️ Entrenando Autoencoder durante 200 épocas en C++...\n" << std::flush;
  for (int epoch = 1; epoch <= 200; ++epoch) {
    optimizer.ZeroGrad();

    // 1. Codificación al espacio latente (Bottleneck z)
    Tensor z = encoder.Forward(X);

    // 2. Decodificación a la reconstrucción X_hat
    Tensor X_hat = decoder.Forward(z);

    // 3. Pérdida de reconstrucción MSE(X_hat, X)
    float loss = criterion.Forward(X_hat, X);

    // 4. Retropropagación a través de Decoder y Encoder
    Tensor dX_hat = criterion.Backward();
    Tensor dz = decoder.Backward(dX_hat);
    encoder.Backward(dz);

    optimizer.Step();

    if (epoch % 50 == 0 || epoch == 200) {
      std::cout << "Época " << epoch << "/200 | Loss Reconstrucción MSE: " << loss << "\n" << std::flush;
    }
  }

  encoder.Save("encoder_model.ns");
  decoder.Save("decoder_model.ns");

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Autoencoder entrenado y verificado exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
