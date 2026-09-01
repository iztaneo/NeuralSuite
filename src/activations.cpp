// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de activations.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "activations.h"

namespace neuralsuite {

Tensor Activation::Forward(const Tensor& input) {
    last_input_ = input;
    Tensor output;
    switch (type_) {
      case ActivationType::kRelu:    ReluForward(input, output); break;
      case ActivationType::kGelu:    GeluForward(input, output); break;
      case ActivationType::kSigmoid: SigmoidForward(input, output); break;
      case ActivationType::kTanh:    TanhForward(input, output); break;
    }
    last_output_ = output;
    return output;
  }

Tensor Activation::Backward(const Tensor& dout) {
    Tensor dx;
    switch (type_) {
      case ActivationType::kRelu:    ReluBackward(dout, last_input_, dx); break;
      case ActivationType::kGelu:    GeluBackward(dout, last_input_, dx); break;
      case ActivationType::kSigmoid: SigmoidBackward(dout, last_output_, dx); break;
      case ActivationType::kTanh:    TanhBackward(dout, last_output_, dx); break;
    }
    return dx;
  }

}  // namespace neuralsuite
