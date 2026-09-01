// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file activations.h
 * @brief Non-linear Activation Layers following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_ACTIVATIONS_H_
#define NEURAL_SUITE_INCLUDE_ACTIVATIONS_H_

#include <vector>
#include "layer.h"

namespace neuralsuite {

enum class ActivationType { kRelu, kGelu, kSigmoid, kTanh };

/**
 * @class Activation
 * @brief Layer wrapper for non-linear activation functions.
 */
class Activation : public Layer {
 public:
  explicit Activation(ActivationType act_type) : type_(act_type) {}

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

 private:
  ActivationType type_;
  Tensor last_input_;
  Tensor last_output_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_ACTIVATIONS_H_
