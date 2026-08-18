// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_autograd.cpp
 * @brief Entrena una red sin escribir ninguna derivada a mano.
 *
 * Las demas demos usan capas cuyo `Backward()` esta escrito explicitamente.
 * Aqui no hay ninguno: se declara el calculo hacia delante y el motor de
 * diferenciacion automatica deduce todos los gradientes.
 *
 * El problema es XOR, que no es separable linealmente y por tanto obliga a que
 * la capa oculta aprenda algo: una red lineal no puede resolverlo.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "autograd.h"
#include "tensor.h"

using namespace neuralsuite;
using namespace neuralsuite::autograd;

namespace {

/** @brief Tensor con valores deterministas, para que la demo sea reproducible. */
Tensor Seeded(const std::vector<int>& shape, float scale, int offset) {
  Tensor t(shape);
  for (size_t i = 0; i < t.TotalSize(); ++i) {
    t[i] = scale * std::sin(1.7f * static_cast<float>(i + offset));
  }
  return t;
}

}  // namespace

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔗 Demostración: entrenamiento con diferenciación automática\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;
  std::cout << "Ninguna derivada esta escrita a mano: solo el paso hacia delante.\n\n"
            << std::flush;

  // Las cuatro combinaciones de XOR.
  Tensor x_data({4, 2});
  const float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (int i = 0; i < 4; ++i) {
    x_data[i * 2] = inputs[i][0];
    x_data[i * 2 + 1] = inputs[i][1];
  }
  Tensor y_data({4, 1});
  y_data[0] = 0.0f; y_data[1] = 1.0f; y_data[2] = 1.0f; y_data[3] = 0.0f;

  auto x = Variable::Create(x_data);
  auto y = Variable::Create(y_data);

  // Pesos de una red 2 -> 8 -> 1. Son las hojas del grafo con gradiente.
  const int hidden = 8;
  auto w1 = Variable::Create(Seeded({2, hidden}, 0.9f, 1), true);
  auto b1 = Variable::Create(Tensor({4, hidden}), true);
  auto w2 = Variable::Create(Seeded({hidden, 1}, 0.9f, 7), true);
  auto b2 = Variable::Create(Tensor({4, 1}), true);

  std::vector<VarPtr> params = {w1, b1, w2, b2};

  const float lr = 0.5f;
  const int epochs = 2000;
  float first_loss = 0.0f, last_loss = 0.0f;

  for (int epoch = 1; epoch <= epochs; ++epoch) {
    for (const VarPtr& p : params) p->ZeroGrad();

    // Paso hacia delante. Esto es todo lo que hay que escribir.
    auto hidden_pre = MatMulVar(x, w1) + b1;
    auto activated = Tanh(hidden_pre);
    auto output = Tanh(MatMulVar(activated, w2) + b2);

    auto diff = output - y;
    auto loss = Mean(diff * diff);   // error cuadratico medio

    Backward(loss);                  // el motor deduce todos los gradientes

    for (const VarPtr& p : params) {
      for (size_t i = 0; i < p->Value().TotalSize(); ++i) {
        p->Value()[i] -= lr * p->Grad()[i];
      }
    }

    if (epoch == 1) first_loss = loss->Value()[0];
    if (epoch == epochs) last_loss = loss->Value()[0];
    if (epoch % 500 == 0 || epoch == 1) {
      std::cout << "Época " << std::setw(4) << epoch
                << " | Loss: " << loss->Value()[0] << "\n" << std::flush;
    }
  }

  // Prediccion final.
  auto activated = Tanh(MatMulVar(x, w1) + b1);
  auto output = Tanh(MatMulVar(activated, w2) + b2);

  std::cout << "\n📊 Predicciones de XOR:\n" << std::flush;
  bool all_correct = true;
  for (int i = 0; i < 4; ++i) {
    const float predicted = output->Value()[i];
    const float expected = y_data[i];
    const bool correct = std::abs(predicted - expected) < 0.4f;
    all_correct = all_correct && correct;
    std::cout << "   " << inputs[i][0] << " XOR " << inputs[i][1]
              << "  ->  " << std::fixed << std::setprecision(3) << predicted
              << "   (esperado " << expected << ")" << (correct ? "  ✅" : "  ❌")
              << "\n" << std::flush;
  }

  std::cout << "\nLoss: " << first_loss << "  ->  " << last_loss << "\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;
  if (all_correct) {
    std::cout << "✅ La red aprendió XOR sin una sola derivada escrita a mano.\n" << std::flush;
  } else {
    std::cerr << "❌ La red no convergió.\n" << std::flush;
    return 1;
  }
  std::cout << "============================================================\n" << std::flush;
  return 0;
}
