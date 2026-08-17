// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file residual.h
 * @brief Residual Block (ResNet Skip / Shortcut Connection y = ReLU(f(x) + x)).
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_RESIDUAL_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_RESIDUAL_H_

#include <memory>
#include <vector>
#include "../activations.h"
#include "../layer.h"

namespace neuralsuite {

/**
 * @class ResidualBlock
 * @brief Implements Residual Skip Connection: y = ReLU(f(x) + x).
 */
class ResidualBlock : public Layer {
 public:
  explicit ResidualBlock(std::shared_ptr<Layer> inner_layer)
      : inner_layer_(inner_layer), relu_(ActivationType::kRelu) {
    Register(inner_layer_.get());
  }

  Tensor Forward(const Tensor& input) override {
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

  Tensor Backward(const Tensor& dout) override {
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

 private:
  std::shared_ptr<Layer> inner_layer_;
  Activation relu_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_RESIDUAL_H_
