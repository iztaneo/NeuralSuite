// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_cnn.cpp
 * @brief CNN 2D Demo following Google C++ Style Guide.
 */

#include <iostream>
#include <vector>
#include "activations.h"
#include "layers/conv2d.h"
#include "layers/linear.h"
#include "layers/maxpool2d.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🖼️ Demostración 2: Red Convolucional (Google C++ Style Guide)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  Tensor X({2, 1, 8, 8});
  X.RandomNormal(0.0f, 1.0f);

  Tensor Y({2});
  Y[0] = 0.0f;
  Y[1] = 1.0f;

  Conv2D conv(1, 4, 3, 1, 0);
  Activation relu(ActivationType::kRelu);
  MaxPool2D pool(2, 2);
  Linear fc(4 * 3 * 3, 2);

  CrossEntropyLoss criterion;

  std::vector<Tensor*> params, grads;
  for (auto p : conv.GetParameters()) params.push_back(p);
  for (auto p : fc.GetParameters()) params.push_back(p);
  for (auto g : conv.GetGradients()) grads.push_back(g);
  for (auto g : fc.GetGradients()) grads.push_back(g);

  AdamW optimizer(params, grads, 0.01f);

  std::cout << "🏋️ Entrenando CNN durante 20 iteraciones...\n" << std::flush;
  for (int epoch = 1; epoch <= 20; ++epoch) {
    optimizer.ZeroGrad();

    Tensor h_conv = conv.Forward(X);
    Tensor h_relu = relu.Forward(h_conv);
    Tensor h_pool = pool.Forward(h_relu);

    Tensor h_flat({2, 4 * 3 * 3});
    std::memcpy(h_flat.Data(), h_pool.Data(), h_pool.TotalSize() * sizeof(float));

    Tensor logits = fc.Forward(h_flat);

    float loss = criterion.Forward(logits, Y);

    Tensor dlogits = criterion.Backward();
    Tensor dh_flat = fc.Backward(dlogits);

    Tensor dh_pool(h_pool.Shape());
    std::memcpy(dh_pool.Data(), dh_flat.Data(), dh_flat.TotalSize() * sizeof(float));

    Tensor dh_relu = pool.Backward(dh_pool);
    Tensor dh_conv = relu.Backward(dh_relu);
    conv.Backward(dh_conv);

    optimizer.Step();

    if (epoch % 5 == 0 || epoch == 1) {
      std::cout << "Época " << epoch << " | Loss Convolucional: " << loss << "\n" << std::flush;
    }
  }

  std::cout << "✅ ¡Entrenamiento CNN completado exitosamente en C++!\n" << std::flush;
  return 0;
}
