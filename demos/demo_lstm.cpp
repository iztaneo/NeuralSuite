// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_lstm.cpp
 * @brief LSTM Recurrent Network Demo following Google C++ Style Guide.
 */

#include <iostream>
#include <vector>
#include "layers/linear.h"
#include "layers/lstm.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔄 Demostración 3: Red Recurrente (LSTM Google Style)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  Tensor X({5, 1, 4});
  X.RandomNormal(0.0f, 1.0f);

  Tensor Y({5, 1, 2});
  Y.Zeros();

  LSTM lstm(4, 8);
  Linear fc(8, 2);

  MSELoss criterion;

  std::vector<Parameter*> params;
  for (Parameter* p : lstm.Parameters()) params.push_back(p);
  for (Parameter* p : fc.Parameters()) params.push_back(p);

  AdamW optimizer(params, 0.01f);

  std::cout << "🏋️ Entrenando LSTM durante 10 iteraciones...\n" << std::flush;
  for (int epoch = 1; epoch <= 10; ++epoch) {
    optimizer.ZeroGrad();

    Tensor h_lstm = lstm.Forward(X);

    Tensor h_2d({5 * 1, 8});
    std::memcpy(h_2d.Data(), h_lstm.Data(), h_lstm.TotalSize() * sizeof(float));

    Tensor logits_2d = fc.Forward(h_2d);

    float loss = criterion.Forward(logits_2d, Y);

    Tensor dlogits = criterion.Backward();
    Tensor dh_2d = fc.Backward(dlogits);

    Tensor dh_lstm(h_lstm.Shape());
    std::memcpy(dh_lstm.Data(), dh_2d.Data(), dh_2d.TotalSize() * sizeof(float));

    lstm.Backward(dh_lstm);
    optimizer.Step();

    std::cout << "Época " << epoch << " | Loss Recurrente LSTM: " << loss << "\n" << std::flush;
  }

  std::cout << "✅ ¡Entrenamiento LSTM completado exitosamente en C++!\n" << std::flush;
  return 0;
}
