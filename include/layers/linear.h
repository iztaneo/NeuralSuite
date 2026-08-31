// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file linear.h
 * @brief Fully Connected Linear Layer with 2D/3D Tensor Support following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_

#include <cstring>
#include <vector>
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class Linear
 * @brief Dense Linear Layer Y = X * W + b for N-Dimensional Tensors.
 */
class Linear : public Layer {
 public:
  /**
   * @brief Capa densa. Por defecto se inicializa con Xavier.
   *
   * `init_std` positivo fuerza en su lugar una normal con esa desviacion. Se
   * usa para reproducir convenciones concretas —GPT-2 fija 0.02 en todas sus
   * densas— pero no debe ser el valor por defecto, y aqui lo era.
   *
   * Un 0.02 fijo no escala con el tamano de la capa, y eso rompe en silencio.
   * Medido sobre el CRNN de OCR, cuya densa final es 128 -> 63: los logits
   * iniciales salian con media 3.6e-03, el softmax quedaba practicamente
   * uniforme, la perdida se clavaba en ln(63) = 4.143 y el gradiente que
   * llegaba a la LSTM era de 8e-07. El modelo no aprendia, y el sintoma no
   * apuntaba a la inicializacion por ninguna parte: parecia que hacian falta
   * mas epocas.
   *
   * Xavier reparte la escala segun cuantas entradas y salidas tiene la capa,
   * que es justo lo que hacen ya `Conv2D` y `LSTM`. `Linear` era la unica que
   * llevaba una constante escrita a mano.
   */
  Linear(int in_features, int out_features, float init_std = 0.0f)
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

  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_
