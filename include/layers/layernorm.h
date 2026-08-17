// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file layernorm.h
 * @brief Layer Normalization Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LAYERNORM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LAYERNORM_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class LayerNormLayer
 * @brief Pre-LN Layer Normalization.
 */
class LayerNormLayer : public Layer {
 public:
  explicit LayerNormLayer(int shape, float epsilon = 1e-5f)
      : normalized_shape_(shape),
        eps_(epsilon),
        gamma_({shape}),
        beta_({shape}),
        dgamma_({shape}),
        dbeta_({shape}) {
    gamma_.Ones();
    beta_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    Tensor output(input.Shape());
    LayerNormForward(input, gamma_, beta_, output, mean_cache_, rstd_cache_, eps_);
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    Tensor dx(last_input_.Shape());
    LayerNormBackward(dout, last_input_, gamma_, mean_cache_, rstd_cache_, dx, dgamma_, dbeta_);
    return dx;
  }

  std::vector<Tensor*> GetParameters() override { return {&gamma_, &beta_}; }
  std::vector<Tensor*> GetGradients() override { return {&dgamma_, &dbeta_}; }

 private:
  int normalized_shape_;
  float eps_;

  Tensor gamma_;
  Tensor beta_;
  Tensor dgamma_;
  Tensor dbeta_;

  Tensor last_input_;
  Tensor mean_cache_;
  Tensor rstd_cache_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LAYERNORM_H_
