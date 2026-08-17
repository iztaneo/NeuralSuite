// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_mlp.cpp
 * @brief MLP Classifier Training Demo following Google C++ Style Guide.
 */

#include <iostream>
#include <vector>
#include "activations.h"
#include "layers/linear.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🧠 Demostración 1: Clasificador MLP (Google C++ Style Guide)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  Tensor X({4, 2});
  X[0] = 0.0f; X[1] = 0.0f;
  X[2] = 0.0f; X[3] = 1.0f;
  X[4] = 1.0f; X[5] = 0.0f;
  X[6] = 1.0f; X[7] = 1.0f;

  Tensor Y({4});
  Y[0] = 0.0f;
  Y[1] = 1.0f;
  Y[2] = 1.0f;
  Y[3] = 0.0f;

  Linear fc1(2, 8);
  Activation relu(ActivationType::kRelu);
  Linear fc2(8, 2);

  CrossEntropyLoss criterion;

  std::vector<Parameter*> params;
  for (Parameter* p : fc1.Parameters()) params.push_back(p);
  for (Parameter* p : fc2.Parameters()) params.push_back(p);

  AdamW optimizer(params, 0.05f);

  std::cout << "🏋️ Entrenando MLP durante 200 iteraciones...\n" << std::flush;
  for (int epoch = 1; epoch <= 200; ++epoch) {
    optimizer.ZeroGrad();

    Tensor h1 = fc1.Forward(X);
    Tensor a1 = relu.Forward(h1);
    Tensor logits = fc2.Forward(a1);

    float loss = criterion.Forward(logits, Y);

    Tensor dlogits = criterion.Backward();
    Tensor da1 = fc2.Backward(dlogits);
    Tensor dh1 = relu.Backward(da1);
    fc1.Backward(dh1);

    optimizer.Step();

    if (epoch % 50 == 0 || epoch == 1) {
      std::cout << "Época " << epoch << " | Loss MLP: " << loss << "\n" << std::flush;
    }
  }

  std::cout << "✅ ¡Entrenamiento MLP completado exitosamente en C++!\n" << std::flush;
  return 0;
}
