// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file rmsnorm.h
 * @brief Normalizacion por la raiz cuadratica media, sin restar la media.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_RMSNORM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_RMSNORM_H_

#include <vector>
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class RMSNormLayer
 * @brief y = x / sqrt(media(x^2) + eps) * gamma
 *
 * Es la normalizacion de LLaMA, Mistral y Gemma, y sustituye a `LayerNorm` en
 * los transformers modernos. La diferencia con `LayerNorm` es que **no resta la
 * media y no tiene sesgo**: solo divide por la raiz cuadratica media.
 *
 * Eso importa por dos razones. Una practica: ahorra una pasada sobre la fila y
 * un parametro por canal. Y otra que se ve en el backward: al no restar la
 * media, el gradiente pierde uno de los tres terminos que tiene el de
 * `LayerNorm`. Queda
 *
 *     dx = (dout * gamma - x * (suma(dout * gamma * x) / (n * ms))) / sqrt(ms)
 *
 * con `ms = media(x^2) + eps`. El termino que sobrevive —el que resta la
 * proyeccion de `x`— es el que hace que el gradiente sea ortogonal a `x`, y
 * omitirlo es el error clasico: la red sigue entrenando, algo peor, y no falla
 * en ninguna prueba que no compruebe el gradiente.
 *
 * `normalized_shape` es el tamano del ultimo eje, que es sobre el que se
 * normaliza; los ejes anteriores se recorren como filas independientes.
 */
class RMSNormLayer : public Layer {
 public:
  explicit RMSNormLayer(int normalized_shape, float epsilon = 1e-5f)
      : normalized_shape_(normalized_shape),
        eps_(epsilon),
        gamma_({normalized_shape}) {
    Register(&gamma_, "gamma");
    // Se inicializa a uno: al empezar, la capa solo normaliza y no reescala.
    for (size_t i = 0; i < gamma_.Value().TotalSize(); ++i) gamma_.Value()[i] = 1.0f;
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Tensor& Gamma() const { return gamma_.Value(); }
  Tensor& Gamma() { return gamma_.Value(); }

 private:
  int normalized_shape_;
  float eps_;

  Parameter gamma_;

  Tensor last_input_;
  Tensor rms_cache_;   // 1/sqrt(media(x^2)+eps) por fila, guardado del forward
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_RMSNORM_H_
