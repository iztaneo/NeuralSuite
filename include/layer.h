// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file layer.h
 * @brief Abstract Base Class for Neural Network Layers following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYER_H_
#define NEURAL_SUITE_INCLUDE_LAYER_H_

#include <vector>
#include "tensor.h"

namespace neuralsuite {

/**
 * @class Layer
 * @brief Abstract Interface for Neural Network Layers (Linear, Conv2D, LSTM, Attention).
 */
class Layer {
 public:
  virtual ~Layer() = default;

  /**
   * @brief Forward pass transformation
   */
  virtual Tensor Forward(const Tensor& input) = 0;

  /**
   * @brief Backward pass gradient propagation
   */
  virtual Tensor Backward(const Tensor& grad_output) = 0;

  /**
   * @brief Returns pointers to trainable parameter tensors
   */
  virtual std::vector<Tensor*> GetParameters() { return {}; }

  /**
   * @brief Returns pointers to gradient tensors
   */
  virtual std::vector<Tensor*> GetGradients() { return {}; }
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYER_H_
