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
    Register(inner_layer_.get(), "inner");
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

 private:
  std::shared_ptr<Layer> inner_layer_;
  Activation relu_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_RESIDUAL_H_
