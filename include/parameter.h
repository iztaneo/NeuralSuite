// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parameter.h
 * @brief Peso entrenable junto a su gradiente, en un solo objeto.
 */

#ifndef NEURAL_SUITE_INCLUDE_PARAMETER_H_
#define NEURAL_SUITE_INCLUDE_PARAMETER_H_

#include <vector>
#include "tensor.h"

namespace neuralsuite {

/**
 * @class Parameter
 * @brief Un tensor entrenable y su gradiente, inseparables.
 *
 * El framework mantenia el valor y el gradiente en dos listas paralelas que el
 * optimizador recorria por indice. Bastaba con que una capa expusiera sus pesos
 * y olvidara exponer sus gradientes para que las listas dejaran de
 * corresponderse y cada peso se actualizara con el gradiente de otra capa; eso
 * es exactamente lo que ocurria en `MultiHeadAttention`.
 *
 * Aqui valor y gradiente nacen juntos y con la misma forma, de modo que no hay
 * dos listas que puedan desincronizarse.
 */
class Parameter {
 public:
  Parameter() = default;

  explicit Parameter(const std::vector<int>& dims) : value_(dims), grad_(dims) {
    grad_.Zeros();
  }

  [[nodiscard]] Tensor& Value() { return value_; }
  [[nodiscard]] const Tensor& Value() const { return value_; }

  [[nodiscard]] Tensor& Grad() { return grad_; }
  [[nodiscard]] const Tensor& Grad() const { return grad_; }

  [[nodiscard]] const std::vector<int>& Shape() const { return value_.Shape(); }
  [[nodiscard]] size_t TotalSize() const { return value_.TotalSize(); }

  void ZeroGrad() { grad_.Zeros(); }

  /** @brief Cambia la forma de ambos a la vez; nunca pueden divergir. */
  void Resize(const std::vector<int>& dims) {
    value_.Resize(dims);
    grad_.Resize(dims);
    grad_.Zeros();
  }

 private:
  Tensor value_;
  Tensor grad_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_PARAMETER_H_
