// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_adaline.cpp
 * @brief Historical ADALINE (ADAptive LINear Element) Demo following Google C++ Style Guide.
 * 
 * ADALINE (Widrow & Hoff, 1960) introduced the Delta Rule / LMS (Least Mean Squares) algorithm
 * using gradient descent on continuous linear outputs before thresholding.
 */

#include <iostream>
#include <vector>
#include "layers/linear.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🏛️ Demostración Histórica: ADALINE (Widrow & Hoff, 1960)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Dataset AND lógico: Entradas X (4 muestras, 2 características)
  Tensor X({4, 2});
  X[0] = -1.0f; X[1] = -1.0f;
  X[2] = -1.0f; X[3] =  1.0f;
  X[4] =  1.0f; X[5] = -1.0f;
  X[6] =  1.0f; X[7] =  1.0f;

  // Salidas esperadas Y (1.0f para verdadero, -1.0f para falso)
  Tensor Y({4, 1});
  Y[0] = -1.0f;
  Y[1] = -1.0f;
  Y[2] = -1.0f;
  Y[3] =  1.0f;

  // ADALINE: Capa Lineal Única (2 entradas -> 1 salida continua z = w^T x + b)
  Linear adaline(2, 1);

  // Función de Pérdida Cuadrática Media (LMS / Least Mean Squares)
  MSELoss criterion;

  std::vector<Tensor*> params = adaline.GetParameters();
  std::vector<Tensor*> grads = adaline.GetGradients();

  AdamW optimizer(params, grads, 0.05f);

  std::cout << "🏋️ Entrenando ADALINE con la Regla Delta (LMS / Gradient Descent)...\n" << std::flush;
  for (int epoch = 1; epoch <= 100; ++epoch) {
    optimizer.ZeroGrad();

    // 1. Salida continua z = w^T x + b (sin umbral previo)
    Tensor z = adaline.Forward(X);

    // 2. Error Cuadrático Medio
    float loss = criterion.Forward(z, Y);

    // 3. Regla Delta / Retropropagación de gradientes
    Tensor dz = criterion.Backward();
    adaline.Backward(dz);
    optimizer.Step();

    if (epoch % 20 == 0 || epoch == 100) {
      std::cout << "Época " << epoch << "/100 | Pérdida LMS (MSE): " << loss << "\n" << std::flush;
    }
  }

  std::cout << "\n🎯 Predicciones de ADALINE tras la Función Umbral (Signo):\n" << std::flush;
  Tensor z_final = adaline.Forward(X);
  for (int i = 0; i < 4; ++i) {
    float val_continua = z_final[i];
    int pred_binaria = (val_continua >= 0.0f) ? 1 : -1;
    int y_esperada = static_cast<int>(Y[i]);
    std::cout << "   - Muestra [" << X[i * 2] << ", " << X[i * 2 + 1] << "] -> Salida Continua: "
              << val_continua << " | Predicción: " << pred_binaria
              << " (Esperado: " << y_esperada << ") "
              << (pred_binaria == y_esperada ? "✅" : "❌") << "\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Entrenamiento de ADALINE completado exitosamente!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
