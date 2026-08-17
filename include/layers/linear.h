// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file linear.h
 * @brief Fully Connected Linear Layer: Y = X * W + b following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class Linear
 * @brief Fully Connected Dense Layer.
 */
class Linear : public Layer {
 public:
  Linear(int in_dim, int out_dim)
      : in_features_(in_dim),
        out_features_(out_dim),
        weight_({in_dim, out_dim}),
        bias_({out_dim}),
        dweight_({in_dim, out_dim}),
        dbias_({out_dim}) {
    weight_.XavierInit(in_dim, out_dim);
    bias_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    Tensor output({batch_size, out_features_});

    MatMul(input, weight_, output);

    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        output[i * out_features_ + j] += bias_[j];
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    int batch_size = last_input_.Shape()[0];
    Tensor dx({batch_size, in_features_});

    Tensor weight_t = Transpose(weight_);
    MatMul(dout, weight_t, dx);

    Tensor input_t = Transpose(last_input_);
    MatMul(input_t, dout, dweight_);

    dbias_.Zeros();
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        dbias_[j] += dout[i * out_features_ + j];
      }
    }
    return dx;
  }

  std::vector<Tensor*> GetParameters() override { return {&weight_, &bias_}; }
  std::vector<Tensor*> GetGradients() override { return {&dweight_, &dbias_}; }

  // Accessors
  [[nodiscard]] Tensor& Weight() { return weight_; }
  [[nodiscard]] Tensor& Bias() { return bias_; }

 private:
  int in_features_;
  int out_features_;

  Tensor weight_;
  Tensor bias_;
  Tensor dweight_;
  Tensor dbias_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_
