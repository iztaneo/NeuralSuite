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
  float Forward(const Tensor& predictions, const Tensor& targets) override {
    last_preds_ = predictions;
    last_targets_ = targets;

    int num_samples = predictions.Shape()[0];
    int num_classes = predictions.Shape()[1];

    float total_loss = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
      int target_cls = static_cast<int>(targets[i]);

      float max_val = predictions[i * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        if (predictions[i * num_classes + c] > max_val) max_val = predictions[i * num_classes + c];
      }

      float sum_exp = 0.0f;
      for (int c = 0; c < num_classes; ++c) {
        sum_exp += std::exp(predictions[i * num_classes + c] - max_val);
      }

      float log_prob = (predictions[i * num_classes + target_cls] - max_val) - std::log(sum_exp);
      total_loss -= log_prob;
    }
    return total_loss / num_samples;
  }


  Tensor Backward() override {
    int num_samples = last_preds_.Shape()[0];
    int num_classes = last_preds_.Shape()[1];

    Tensor dlogits(last_preds_.Shape());

    for (int i = 0; i < num_samples; ++i) {
      int target_cls = static_cast<int>(last_targets_[i]);

      float max_val = last_preds_[i * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        if (last_preds_[i * num_classes + c] > max_val) max_val = last_preds_[i * num_classes + c];
      }

      float sum = 0.0f;
      for (int c = 0; c < num_classes; ++c) sum += std::exp(last_preds_[i * num_classes + c] - max_val);

      for (int c = 0; c < num_classes; ++c) {
        float prob = std::exp(last_preds_[i * num_classes + c] - max_val) / sum;
        float target_val = (c == target_cls) ? 1.0f : 0.0f;
        dlogits[i * num_classes + c] = (prob - target_val) / num_samples;
      }
    }
    return dlogits;
  }

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
  float Forward(const Tensor& predictions, const Tensor& targets) override {
    last_preds_ = predictions;
    last_targets_ = targets;
    size_t sz = predictions.TotalSize();

    float loss = 0.0f;
    for (size_t i = 0; i < sz; ++i) {
      float diff = predictions[i] - targets[i];
      loss += diff * diff;
    }
    return loss / sz;
  }

  Tensor Backward() override {
    size_t sz = last_preds_.TotalSize();
    Tensor dpreds(last_preds_.Shape());
    for (size_t i = 0; i < sz; ++i) {
      dpreds[i] = 2.0f * (last_preds_[i] - last_targets_[i]) / sz;
    }
    return dpreds;
  }

 private:
  Tensor last_preds_;
  Tensor last_targets_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LOSSES_H_
