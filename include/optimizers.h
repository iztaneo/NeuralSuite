// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file optimizers.h
 * @brief Optimizers (AdamW, SGD) following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_
#define NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "parameter.h"
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
 * @brief Comprueba que parámetros y gradientes formen pares válidos.
 *
 * Step() recorre ambas listas por índice, así que una discrepancia de tamaño o
 * de forma haría que un parámetro se actualizara con el gradiente de otra capa
 * y, cuando el gradiente tiene menos elementos, que se leyera fuera de su
 * memoria. Es preferible fallar aquí y de inmediato a entrenar un modelo
 * corrupto durante horas.
 */
inline void ValidateParamGradPairs(const std::vector<Tensor*>& params,
                                   const std::vector<Tensor*>& grads) {
  if (params.size() != grads.size()) {
    throw std::invalid_argument(
        "Optimizador: " + std::to_string(params.size()) + " parametros frente a " +
        std::to_string(grads.size()) +
        " gradientes. Alguna capa expone GetParameters() pero no GetGradients().");
  }
  for (size_t i = 0; i < params.size(); ++i) {
    if (params[i] == nullptr || grads[i] == nullptr) {
      throw std::invalid_argument("Optimizador: puntero nulo en el indice " +
                                  std::to_string(i) + ".");
    }
    if (params[i]->Shape() != grads[i]->Shape()) {
      throw std::invalid_argument(
          "Optimizador: el parametro y el gradiente del indice " + std::to_string(i) +
          " tienen formas distintas.");
    }
  }
}

/**
 * @class SGD
 * @brief Stochastic Gradient Descent with Momentum.
 */
class SGD : public Optimizer {
 public:
  /**
   * @brief Construccion recomendada: una sola lista de parametros.
   *
   * Cada Parameter lleva su gradiente dentro, asi que no hay dos listas que
   * puedan dejar de corresponderse.
   */
  explicit SGD(const std::vector<Parameter*>& parameters, float learning_rate = 0.01f,
               float mom = 0.9f)
      : lr_(learning_rate), momentum_(mom) {
    for (Parameter* p : parameters) {
      if (p == nullptr) throw std::invalid_argument("SGD: parametro nulo.");
      params_.push_back(&p->Value());
      grads_.push_back(&p->Grad());
    }
    ValidateParamGradPairs(params_, grads_);
    for (auto p : params_) {
      Tensor v(p->Shape());
      v.Zeros();
      velocities_.push_back(v);
    }
  }

  /** @brief Forma heredada, con dos listas que se validan al construir. */
  SGD(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
      float learning_rate = 0.01f, float mom = 0.9f)
      : params_(parameters), grads_(gradients), lr_(learning_rate), momentum_(mom) {
    ValidateParamGradPairs(params_, grads_);
    for (auto p : params_) {
      Tensor v(p->Shape());
      v.Zeros();
      velocities_.push_back(v);
    }
  }

  void Step() override;

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
 * @struct ParamGroup
 * @brief Conjunto de parametros que comparten su propio weight decay.
 *
 * AdamW deducia el decay del rango del tensor: se lo aplicaba a todo lo que
 * tuviera dos ejes o mas. Funciona por casualidad con la convencion habitual
 * —matrices si, sesgos y LayerNorm no— pero es una inferencia, no una
 * declaracion: una capa con un parametro 2D que no deba decaer recibiria el
 * tratamiento equivocado sin que nada lo indique.
 */
struct ParamGroup {
  std::vector<Parameter*> params;
  float weight_decay = 0.0f;
};

/**
 * @class AdamW
 * @brief AdamW Optimizer with Decoupled Weight Decay.
 */
class AdamW : public Optimizer {
 public:
  /**
   * @brief Construccion recomendada: una sola lista de parametros.
   */
  explicit AdamW(const std::vector<Parameter*>& parameters, float learning_rate = 1e-3f,
                 float b1 = 0.9f, float b2 = 0.95f, float epsilon = 1e-8f, float wd = 0.01f)
      : lr_(learning_rate), beta1_(b1), beta2_(b2), eps_(epsilon), weight_decay_(wd),
        step_count_(0) {
    for (Parameter* p : parameters) {
      if (p == nullptr) throw std::invalid_argument("AdamW: parametro nulo.");
      params_.push_back(&p->Value());
      grads_.push_back(&p->Grad());
    }
    ValidateParamGradPairs(params_, grads_);
    for (auto p : params_) {
      Tensor m_i(p->Shape()); m_i.Zeros(); m_.push_back(m_i);
      Tensor v_i(p->Shape()); v_i.Zeros(); v_.push_back(v_i);
    }
  }

  /** @brief Forma heredada, con dos listas que se validan al construir. */
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
    ValidateParamGradPairs(params_, grads_);
    for (auto p : params_) {
      Tensor m_i(p->Shape()); m_i.Zeros(); m_.push_back(m_i);
      Tensor v_i(p->Shape()); v_i.Zeros(); v_.push_back(v_i);
    }
  }

  /**
   * @brief Construccion por grupos, cada uno con su weight decay declarado.
   */
  explicit AdamW(const std::vector<ParamGroup>& groups, float learning_rate = 1e-3f,
                 float b1 = 0.9f, float b2 = 0.95f, float epsilon = 1e-8f)
      : lr_(learning_rate), beta1_(b1), beta2_(b2), eps_(epsilon),
        weight_decay_(0.0f), step_count_(0), use_groups_(true) {
    for (const ParamGroup& group : groups) {
      for (Parameter* p : group.params) {
        if (p == nullptr) throw std::invalid_argument("AdamW: parametro nulo en un grupo.");
        params_.push_back(&p->Value());
        grads_.push_back(&p->Grad());
        decays_.push_back(group.weight_decay);
      }
    }
    ValidateParamGradPairs(params_, grads_);
    for (auto p : params_) {
      Tensor m_i(p->Shape()); m_i.Zeros(); m_.push_back(m_i);
      Tensor v_i(p->Shape()); v_i.Zeros(); v_.push_back(v_i);
    }
  }

  void SetLearningRate(float lr) { lr_ = lr; }

  void Step() override;

  void ZeroGrad() override {
    for (auto g : grads_) g->Zeros();
  }

 private:
  std::vector<Tensor*> params_;
  std::vector<Tensor*> grads_;
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;
  std::vector<float> decays_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int step_count_;
  bool use_groups_ = false;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_OPTIMIZERS_H_
