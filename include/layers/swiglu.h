// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file swiglu.h
 * @brief Bloque feed-forward con puerta, el de LLaMA y Mistral.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_SWIGLU_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_SWIGLU_H_

#include <stdexcept>
#include <string>

#include "../layer.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class SwiGLU
 * @brief `(SiLU(x·Wg) ⊙ x·Wu) · Wd`
 *
 * Sustituye al feed-forward clasico `Linear -> GELU -> Linear`. La diferencia
 * es la **puerta**: en vez de aplicar una activacion y ya, se calculan dos
 * proyecciones y una regula a la otra multiplicandola elemento a elemento. Eso
 * deja que la red decida por canal cuanto deja pasar, en vez de aplicar la
 * misma curva a todos.
 *
 * El precio son **tres matrices en vez de dos**. Para que el numero de
 * parametros no crezca, la convencion —la de LLaMA— es usar un tamano oculto
 * de `8/3` veces la dimension en vez de `4`, que es lo que deja los dos bloques
 * mas o menos iguales: `3 · (8/3) = 8`, frente a `2 · 4 = 8`. Aqui el tamano
 * oculto se pasa explicito para no esconder esa cuenta.
 *
 * El error facil es multiplicar al reves: aplicar `SiLU` a la proyeccion de
 * arriba en vez de a la puerta. Los dos caminos dan numeros del mismo orden y
 * la red entrena, algo peor.
 */
class SwiGLU : public Layer {
 public:
  SwiGLU(int dim, int oculto)
      : dim_(dim),
        oculto_(oculto),
        w_gate_(dim, oculto),
        w_up_(dim, oculto),
        w_down_(oculto, dim) {
    if (dim <= 0 || oculto <= 0) {
      throw std::invalid_argument("SwiGLU: las dimensiones deben ser positivas.");
    }
    Register(&w_gate_, "w_gate");
    Register(&w_up_, "w_up");
    Register(&w_down_, "w_down");
  }

  Tensor Forward(const Tensor& input) override;
  Tensor Backward(const Tensor& dout) override;

  Linear& Gate() { return w_gate_; }
  Linear& Up() { return w_up_; }
  Linear& Down() { return w_down_; }

  /** @brief El oculto que usa LLaMA: 8/3 de la dimension, multiplo de 256. */
  static int OcultoLlama(int dim, int multiplo = 256) {
    const int base = (8 * dim) / 3;
    return ((base + multiplo - 1) / multiplo) * multiplo;
  }

 private:
  int dim_;
  int oculto_;

  Linear w_gate_, w_up_, w_down_;

  Tensor g_lineal_;   // x·Wg antes de la activacion; el backward de SiLU lo pide
  Tensor g_act_;      // SiLU(x·Wg)
  Tensor u_;          // x·Wu
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_SWIGLU_H_
