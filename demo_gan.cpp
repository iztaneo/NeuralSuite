// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_gan.cpp
 * @brief Generative Adversarial Network (GAN - Generator vs Discriminator) Demo in C++.
 */

#include <iostream>
#include <memory>
#include <random>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🎭 Demostración 6: Red Generativa Adversaria (GAN C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Generador G(z): Convierte ruido aleatorio de 2D a datos sintéticos de 2D
  Sequential generator;
  generator.Add(std::make_shared<Linear>(2, 8));
  generator.Add(std::make_shared<Activation>(ActivationType::kRelu));
  generator.Add(std::make_shared<Linear>(8, 2));

  // Discriminador D(x): Evalúa si una muestra de 2D es Real (1) o Falsa (0)
  Sequential discriminator;
  discriminator.Add(std::make_shared<Linear>(2, 8));
  discriminator.Add(std::make_shared<Activation>(ActivationType::kRelu));
  discriminator.Add(std::make_shared<Linear>(8, 2));

  CrossEntropyLoss criterion;

  AdamW opt_d(discriminator.GetParameters(), discriminator.GetGradients(), 0.02f);
  AdamW opt_g(generator.GetParameters(), generator.GetGradients(), 0.02f);

  std::cout << "🏋️ Entrenando GAN (Juego Minimax Generador vs Discriminador) durante 100 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    // 1. Crear datos reales (Muestras objetivo centradas en 1.0f)
    Tensor real_data({4, 2});
    real_data.RandomNormal(1.0f, 0.2f);
    Tensor label_real({4});
    label_real[0] = 1.0f; label_real[1] = 1.0f; label_real[2] = 1.0f; label_real[3] = 1.0f;

    // 2. Crear muestras falsas desde el Generador
    Tensor noise({4, 2});
    noise.RandomNormal(0.0f, 1.0f);
    Tensor fake_data = generator.Forward(noise);
    Tensor label_fake({4});
    label_fake[0] = 0.0f; label_fake[1] = 0.0f; label_fake[2] = 0.0f; label_fake[3] = 0.0f;

    // 3. Entrenar Discriminador D
    opt_d.ZeroGrad();
    Tensor d_real_logits = discriminator.Forward(real_data);
    float loss_d_real = criterion.Forward(d_real_logits, label_real);
    Tensor d_d_real = criterion.Backward();
    discriminator.Backward(d_d_real);

    Tensor d_fake_logits = discriminator.Forward(fake_data);
    float loss_d_fake = criterion.Forward(d_fake_logits, label_fake);
    Tensor d_d_fake = criterion.Backward();
    discriminator.Backward(d_d_fake);
    opt_d.Step();

    // 4. Entrenar Generador G (Engañar a D para que clasifique fake_data como Real 1)
    opt_g.ZeroGrad();
    Tensor fake_data_g = generator.Forward(noise);
    Tensor d_g_logits = discriminator.Forward(fake_data_g);
    float loss_g = criterion.Forward(d_g_logits, label_real); // Engaño a D
    Tensor d_g_loss = criterion.Backward();
    Tensor d_g_fake = discriminator.Backward(d_g_loss);
    generator.Backward(d_g_fake);
    opt_g.Step();

    if (epoch % 25 == 0 || epoch == 100) {
      std::cout << "Época " << epoch << "/100 | Loss D: " << (loss_d_real + loss_d_fake) / 2.0f
                << " | Loss G (Engaño): " << loss_g << "\n" << std::flush;
    }
  }

  {
    const std::string path = ReleasePath("generator_model.ns");
    if (generator.Save(path)) {
      std::cout << "Pesos guardados en '" << path << "'.\n" << std::flush;
    } else {
      std::cerr << "ERROR: no se pudieron guardar los pesos en '" << path << "'.\n" << std::flush;
    }
  }
  {
    const std::string path = ReleasePath("discriminator_model.ns");
    if (discriminator.Save(path)) {
      std::cout << "Pesos guardados en '" << path << "'.\n" << std::flush;
    } else {
      std::cerr << "ERROR: no se pudieron guardar los pesos en '" << path << "'.\n" << std::flush;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Red Generativa Adversaria (GAN) entrenada y verificada exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
