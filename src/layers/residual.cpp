// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/residual.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/residual.h"

namespace neuralsuite {

Tensor ResidualBlock::Forward(const Tensor& input) {
    last_input_ = input;
    Tensor fx = inner_layer_->Forward(input);

    // Suma residual shortcut: y = f(x) + x
    Tensor sum(fx.Shape());
    size_t sz = fx.TotalSize();
    for (size_t i = 0; i < sz; ++i) {
      sum[i] = fx[i] + input[i];
    }

    return relu_.Forward(sum);
  }

Tensor ResidualBlock::Backward(const Tensor& dout) {
    Tensor dsum = relu_.Backward(dout);

    // Gradiente hacia la función interna f(x)
    Tensor dfx = inner_layer_->Backward(dsum);

    // Gradiente hacia la conexión de salto shortcut x: dinput = dfx + dsum
    Tensor dinput(dfx.Shape());
    size_t sz = dfx.TotalSize();
    for (size_t i = 0; i < sz; ++i) {
      dinput[i] = dfx[i] + dsum[i];
    }

    return dinput;
  }

}  // namespace neuralsuite
