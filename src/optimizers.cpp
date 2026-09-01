// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de optimizers.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "optimizers.h"

namespace neuralsuite {

void SGD::Step() {
    for (size_t i = 0; i < params_.size(); ++i) {
      size_t sz = params_[i]->TotalSize();
      for (size_t k = 0; k < sz; ++k) {
        velocities_[i][k] = momentum_ * velocities_[i][k] + lr_ * grads_[i]->operator[](k);
        (*params_[i])[k] -= velocities_[i][k];
      }
    }
  }

void AdamW::Step() {

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

        // Con grupos el decay viene declarado; sin ellos se mantiene la
        // heuristica anterior por compatibilidad.
        const float decay = use_groups_ ? decays_[i]
                                        : (params_[i]->Shape().size() >= 2 ? weight_decay_ : 0.0f);
        if (decay > 0.0f) {
          (*params_[i])[k] -= lr_ * decay * (*params_[i])[k];
        }

        (*params_[i])[k] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
      }
    }
  }

}  // namespace neuralsuite
