// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file losses.h
 * @brief Loss Functions (CrossEntropyLoss, MSELoss) following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LOSSES_H_
#define NEURAL_SUITE_INCLUDE_LOSSES_H_

#include <cmath>
#include "tensor.h"

namespace neuralsuite {

/**
 * @class Loss
 * @brief Abstract Base Class for Loss Functions.
 */
class Loss {
 public:
  virtual ~Loss() = default;
  virtual float Forward(const Tensor& predictions, const Tensor& targets) = 0;
  virtual Tensor Backward() = 0;
};

/**
 * @class CrossEntropyLoss
 * @brief Categorical Cross-Entropy Loss with Softmax.
 */
class CrossEntropyLoss : public Loss {
 public:
  float Forward(const Tensor& predictions, const Tensor& targets) override;


  Tensor Backward() override;

 private:
  Tensor last_preds_;
  Tensor last_targets_;
};

/**
 * @class MSELoss
 * @brief Mean Squared Error Loss for Regression.
 */
class MSELoss : public Loss {
 public:
  float Forward(const Tensor& predictions, const Tensor& targets) override;

  Tensor Backward() override;

 private:
  Tensor last_preds_;
  Tensor last_targets_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LOSSES_H_
