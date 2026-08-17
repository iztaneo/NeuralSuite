// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file network.h
 * @brief Sequential Layer Container following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_NETWORK_H_
#define NEURAL_SUITE_INCLUDE_NETWORK_H_

#include <memory>
#include <vector>
#include "layer.h"

namespace neuralsuite {

/**
 * @class Sequential
 * @brief Sequential Container for chaining layers.
 */
class Sequential {
 public:
  void Add(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
  }

  Tensor Forward(const Tensor& input) {
    Tensor current = input;
    for (auto& layer : layers_) {
      current = layer->Forward(current);
    }
    return current;
  }

  Tensor Backward(const Tensor& grad_output) {
    Tensor current_grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      current_grad = (*it)->Backward(current_grad);
    }
    return current_grad;
  }

  std::vector<Tensor*> GetParameters() {
    std::vector<Tensor*> all_params;
    for (auto& layer : layers_) {
      auto p = layer->GetParameters();
      all_params.insert(all_params.end(), p.begin(), p.end());
    }
    return all_params;
  }

  std::vector<Tensor*> GetGradients() {
    std::vector<Tensor*> all_grads;
    for (auto& layer : layers_) {
      auto g = layer->GetGradients();
      all_grads.insert(all_grads.end(), g.begin(), g.end());
    }
    return all_grads;
  }

 private:
  std::vector<std::shared_ptr<Layer>> layers_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_NETWORK_H_
