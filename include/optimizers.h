// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file optimizers.h
 * @brief Optimizers (AdamW, SGD) following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_
#define NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_

#include <cmath>
#include <vector>
#include "tensor.h"

namespace neuralsuite {

/**
 * @class Optimizer
 * @brief Abstract Base Class for Optimizers.
 */
class Optimizer {
 public:
  virtual ~Optimizer() = default;
  virtual void Step() = 0;
  virtual void ZeroGrad() = 0;
};

/**
 * @class SGD
 * @brief Stochastic Gradient Descent with Momentum.
 */
class SGD : public Optimizer {
 public:
  SGD(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
      float learning_rate = 0.01f, float mom = 0.9f)
      : params_(parameters), grads_(gradients), lr_(learning_rate), momentum_(mom) {
    for (auto p : params_) {
      Tensor v(p->Shape());
      v.Zeros();
      velocities_.push_back(v);
    }
  }

  void Step() override {
    for (size_t i = 0; i < params_.size(); ++i) {
      size_t sz = params_[i]->TotalSize();
      for (size_t k = 0; k < sz; ++k) {
        velocities_[i][k] = momentum_ * velocities_[i][k] + lr_ * grads_[i]->operator[](k);
        (*params_[i])[k] -= velocities_[i][k];
      }
    }
  }

  void ZeroGrad() override {
    for (auto g : grads_) g->Zeros();
  }

 private:
  std::vector<Tensor*> params_;
  std::vector<Tensor*> grads_;
  std::vector<Tensor> velocities_;
  float lr_;
  float momentum_;
};

/**
 * @class AdamW
 * @brief AdamW Optimizer with Decoupled Weight Decay.
 */
class AdamW : public Optimizer {
 public:
  AdamW(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
        float learning_rate = 1e-3f, float b1 = 0.9f, float b2 = 0.95f, float epsilon = 1e-8f, float wd = 0.01f)
      : params_(parameters),
        grads_(gradients),
        lr_(learning_rate),
        beta1_(b1),
        beta2_(b2),
        eps_(epsilon),
        weight_decay_(wd),
        step_count_(0) {
    for (auto p : params_) {
      Tensor m_i(p->Shape()); m_i.Zeros(); m_.push_back(m_i);
      Tensor v_i(p->Shape()); v_i.Zeros(); v_.push_back(v_i);
    }
  }

  void SetLearningRate(float lr) { lr_ = lr; }

  void Step() override {

    step_count_++;
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t i = 0; i < params_.size(); ++i) {
      size_t sz = params_[i]->TotalSize();
      for (size_t k = 0; k < sz; ++k) {
        float g = (*grads_[i])[k];

        m_[i][k] = beta1_ * m_[i][k] + (1.0f - beta1_) * g;
        v_[i][k] = beta2_ * v_[i][k] + (1.0f - beta2_) * g * g;

        float m_hat = m_[i][k] / bias_correction1;
        float v_hat = v_[i][k] / bias_correction2;

        if (params_[i]->Shape().size() >= 2 && weight_decay_ > 0.0f) {
          (*params_[i])[k] -= lr_ * weight_decay_ * (*params_[i])[k];
        }

        (*params_[i])[k] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
      }
    }
  }

  void ZeroGrad() override {
    for (auto g : grads_) g->Zeros();
  }

 private:
  std::vector<Tensor*> params_;
  std::vector<Tensor*> grads_;
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int step_count_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_
