// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file layer.h
 * @brief Abstract Base Class for Neural Network Layers following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYER_H_
#define NEURAL_SUITE_INCLUDE_LAYER_H_

#include <vector>
#include "module.h"
#include "tensor.h"

namespace neuralsuite {

/**
 * @class Layer
 * @brief Abstract Interface for Neural Network Layers (Linear, Conv2D, LSTM, Attention).
 *
 * Una capa es un `Module` con paso hacia delante y hacia atras. Declara sus
 * parametros en el constructor con `Register()`, y de esa unica declaracion
 * salen tanto la lista de pesos como la de gradientes.
 */
class Layer : public Module {
 public:
  /**
   * @brief Forward pass transformation
   */
  virtual Tensor Forward(const Tensor& input) = 0;

  /**
   * @brief Backward pass gradient propagation
   */
  virtual Tensor Backward(const Tensor& grad_output) = 0;

  /**
   * @brief Punteros a los tensores de pesos entrenables.
   *
   * No es virtual y no debe sobrescribirse: se deriva de `Parameters()`, igual
   * que `GetGradients()`. Que ambas listas salgan de la misma fuente es lo que
   * garantiza que se correspondan elemento a elemento. Cuando eran dos metodos
   * virtuales independientes, una capa podia implementar uno y olvidar el otro:
   * asi fue como los pesos de la atencion acabaron recibiendo los gradientes de
   * otras capas.
   */
  [[nodiscard]] std::vector<Tensor*> GetParameters() {
    std::vector<Tensor*> out;
    for (Parameter* p : Parameters()) out.push_back(&p->Value());
    return out;
  }

  /** @brief Punteros a los gradientes, en el mismo orden que GetParameters(). */
  [[nodiscard]] std::vector<Tensor*> GetGradients() {
    std::vector<Tensor*> out;
    for (Parameter* p : Parameters()) out.push_back(&p->Grad());
    return out;
  }
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYER_H_
