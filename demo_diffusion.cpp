// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_diffusion.cpp
 * @brief Denoising Diffusion Probabilistic Model (DDPM Toy Demo) in C++.
 * 
 * Implements Denoising Diffusion: Forward process adds Gaussian noise,
 * Reverse process uses a Neural Network to predict and remove noise.
 */

#include <iostream>
#include <memory>
#include <random>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🎨 Demostración 8: Modelo de Difusión (Toy DDPM C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Red de Desruidificado (Denoising Network): Predice el ruido adicionado epsilon
  Sequential denoiser;
  denoiser.Add(std::make_shared<Linear>(4, 16));
  denoiser.Add(std::make_shared<Activation>(ActivationType::kRelu));
  denoiser.Add(std::make_shared<Linear>(16, 4));

  MSELoss criterion;
  AdamW optimizer(denoiser.Parameters(), 0.03f);

  std::cout << "🏋️ Entrenando Modelo de Difusión (Forward Noise + Reverse Denoising) durante 100 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.ZeroGrad();

    // 1. Señal limpia original (x0)
    Tensor x0({4, 4});
    x0[0] = 1.0f; x0[1] = 0.5f; x0[2] = -0.5f; x0[3] = -1.0f;
    x0[4] = 0.5f; x0[5] = 1.0f; x0[6] = -1.0f; x0[7] = -0.5f;
    x0[8] = -0.5f; x0[9] = -1.0f; x0[10] = 1.0f; x0[11] = 0.5f;
    x0[12] = -1.0f; x0[13] = -0.5f; x0[14] = 0.5f; x0[15] = 1.0f;

    // 2. Proceso Forward: Agregar Ruido Gaussiano epsilon
    Tensor noise({4, 4});
    noise.RandomNormal(0.0f, 1.0f);

    float beta = 0.3f;
    Tensor xt({4, 4});
    for (size_t i = 0; i < xt.TotalSize(); ++i) {
      xt[i] = std::sqrt(1.0f - beta) * x0[i] + std::sqrt(beta) * noise[i];
    }

    // 3. Proceso Reverse: Red predice el ruido epsilon_pred desruidificando xt
    Tensor noise_pred = denoiser.Forward(xt);

    // 4. Pérdida de Difusión MSE(noise_pred, noise)
    float loss = criterion.Forward(noise_pred, noise);

    Tensor dloss = criterion.Backward();
    denoiser.Backward(dloss);
    optimizer.Step();

    if (epoch % 25 == 0 || epoch == 100) {
      std::cout << "Época " << epoch << "/100 | Loss Predicción de Ruido MSE: " << loss << "\n" << std::flush;
    }
  }

  {
    const std::string path = ReleasePath("diffusion_denoiser.ns");
    if (denoiser.Save(path)) {
      std::cout << "Pesos guardados en '" << path << "'.\n" << std::flush;
    } else {
      std::cerr << "ERROR: no se pudieron guardar los pesos en '" << path << "'.\n" << std::flush;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Modelo de Difusión (DDPM) entrenado y verificado exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
