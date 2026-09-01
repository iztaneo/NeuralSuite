// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/layernorm.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/layernorm.h"

namespace neuralsuite {

Tensor LayerNormLayer::Forward(const Tensor& input) {
    last_input_ = input;
    Tensor output(input.Shape());
    LayerNormForward(input, gamma_.Value(), beta_.Value(), output, mean_cache_, rstd_cache_, eps_);
    return output;
  }

Tensor LayerNormLayer::Backward(const Tensor& dout) {
    Tensor dx(last_input_.Shape());
    LayerNormBackward(dout, last_input_, gamma_.Value(), mean_cache_, rstd_cache_, dx,
                      gamma_.Grad(), beta_.Grad());
    return dx;
  }

}  // namespace neuralsuite
