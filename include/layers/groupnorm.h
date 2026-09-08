// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file groupnorm.h
 * @brief Normalizacion por grupos de canales, independiente del tamano del lote.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_GROUPNORM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_GROUPNORM_H_

#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class GroupNormLayer
 * @brief Normaliza cada grupo de canales por su media y varianza.
 *
 * Es la normalizacion de las U-Net de difusion, y la razon de usarla en vez de
 * `BatchNorm` es que **no mira el lote**: cada ejemplo se normaliza con sus
 * propias estadisticas. En difusion los lotes son pequenos —la memoria se va en
 * los mapas de activacion— y `BatchNorm` con lotes pequenos estima mal la
 * varianza y mete ruido que depende de con quien te toco compartir lote.
 *
 * Se divide `C` en `G` grupos y se normaliza sobre `(C/G, alto, ancho)`, es
 * decir sobre varios canales a la vez. Eso la distingue de `LayerNorm` —que
 * normalizaria sobre todos los canales— y de `InstanceNorm`, que seria un grupo
 * por canal; las dos son casos extremos de esta.
 *
 * `gamma` y `beta` son **por canal**, no por grupo: hay `C` de cada uno. Es una
 * asimetria facil de equivocar, porque las estadisticas son por grupo pero la
 * escala es por canal.
 *
 * La entrada es `[N, C, ...]`: cualquier numero de ejes espaciales despues del
 * canal, de modo que sirve igual para `[N, C, H, W]` que para `[N, C, L]`.
 */
class GroupNormLayer : public Layer {
 public:
  GroupNormLayer(int num_groups, int num_channels, float epsilon = 1e-5f)
      : grupos_(num_groups),
        canales_(num_channels),
        eps_(epsilon),
        gamma_({num_channels}),
        beta_({num_channels}) {
    if (num_groups <= 0 || num_channels <= 0 || num_channels % num_groups != 0) {
      throw std::invalid_argument(
          "GroupNorm: " + std::to_string(num_channels) + " canales no se reparten "
          "en " + std::to_string(num_groups) + " grupos iguales.");
    }
    Register(&gamma_, "gamma");
    Register(&beta_, "beta");
    for (size_t i = 0; i < gamma_.Value().TotalSize(); ++i) gamma_.Value()[i] = 1.0f;
    beta_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Tensor& Gamma() const { return gamma_.Value(); }
  Tensor& Gamma() { return gamma_.Value(); }
  [[nodiscard]] const Tensor& Beta() const { return beta_.Value(); }
  Tensor& Beta() { return beta_.Value(); }

 private:
  int grupos_;
  int canales_;
  float eps_;

  Parameter gamma_;
  Parameter beta_;

  Tensor last_input_;
  Tensor media_;    // una por (ejemplo, grupo)
  Tensor rstd_;     // 1/sqrt(var+eps) por (ejemplo, grupo)
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_GROUPNORM_H_
