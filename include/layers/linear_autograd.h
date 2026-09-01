// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file linear_autograd.h
 * @brief La misma capa densa, pero derivada por el grafo en vez de a mano.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_AUTOGRAD_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_AUTOGRAD_H_

#include <vector>
#include "../autograd.h"
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class LinearAutograd
 * @brief Y = X * W + b, con el gradiente obtenido recorriendo el grafo.
 *
 * Intercambiable con `Linear`: mismo constructor, misma interfaz, mismos
 * parametros en el mismo orden. La diferencia esta en `Backward`, que aqui
 * nadie escribio — sale de encadenar las derivadas de `MatMulVar` y `Add`.
 *
 * Existe por la misma razon que `Conv2DReference`, `LSTMReference` y
 * `MultiHeadAttentionReference`: dos caminos independientes al mismo numero,
 * y una prueba que los enfrenta. Aqui la independencia es mas fuerte que en
 * esos tres, porque no es una version literal de la misma formula: es otra
 * forma de llegar a ella. Los dos defectos P0 del proyecto fueron gradientes
 * mal escritos a mano, y un backward que nadie escribio no puede tenerlos.
 *
 * Lo que se paga a cambio: el grafo reserva los tensores intermedios de cada
 * paso hacia adelante. Para entrenar de verdad sigue siendo mejor `Linear`;
 * esta es la que dice si `Linear` acierta.
 */
class LinearAutograd : public Layer {
 public:
  LinearAutograd(int in_features, int out_features, float init_std = 0.0f)
      : in_features_(in_features),
        out_features_(out_features),
        weight_({in_features, out_features}),
        bias_({out_features}) {
    Register(&weight_, "weight");
    Register(&bias_, "bias");
    if (init_std > 0.0f) {
      weight_.Value().RandomNormal(0.0f, init_std);
    } else {
      weight_.Value().XavierInit(in_features, out_features);
    }
    bias_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Tensor& Weight() const { return weight_.Value(); }
  Tensor& Weight() { return weight_.Value(); }
  [[nodiscard]] const Tensor& Bias() const { return bias_.Value(); }
  Tensor& Bias() { return bias_.Value(); }

 private:
  int in_features_;
  int out_features_;

  Parameter weight_;
  Parameter bias_;

  // El grafo del ultimo paso hacia adelante, que `Backward` recorre.
  autograd::VarPtr entrada_;
  autograd::VarPtr peso_;
  autograd::VarPtr sesgo_;
  autograd::VarPtr salida_;
  std::vector<int> forma_entrada_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_AUTOGRAD_H_
