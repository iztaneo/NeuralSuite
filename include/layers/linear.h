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

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int num_dims = input.Shape().size();
    int in_dim = input.Shape()[num_dims - 1];
    int N = input.TotalSize() / in_dim;

    // Aplanar a 2D es solo una relectura de los ejes: la vista comparte la
    // memoria del original en vez de reservar un tensor y copiarlo entero.
    const Tensor input_2d = input.View({N, in_dim});

    Tensor output_2d({N, out_features_});
    MatMul(input_2d, weight_.Value(), output_2d);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        output_2d[i * out_features_ + j] += bias_.Value()[j];
      }
    }

    // La salida ya tiene los datos correctos; solo hay que devolverla con el
    // rango del tensor de entrada.
    std::vector<int> out_shape = input.Shape();
    out_shape[num_dims - 1] = out_features_;
    output_2d.Reshape(out_shape);
    return output_2d;
  }

  Tensor Backward(const Tensor& dout) override {
    int num_dims = last_input_.Shape().size();
    int in_dim = last_input_.Shape()[num_dims - 1];
    int N = last_input_.TotalSize() / in_dim;

    // Ambas reinterpretaciones son vistas: no se copia nada.
    const Tensor dout_2d = dout.View({N, out_features_});
    const Tensor input_2d = last_input_.View({N, in_dim});

    // dx_2d = dout_2d * weight_^T  ([N, out] * [out, in] -> [N, in])
    Tensor dx_2d({N, in_dim});
    Tensor weight_t = Transpose(weight_.Value());
    MatMul(dout_2d, weight_t, dx_2d);

    // dweight = input_2d^T * dout_2d ([in, N] * [N, out] -> [in, out])
    Tensor input_t = Transpose(input_2d);
    MatMul(input_t, dout_2d, weight_.Grad());

    // dbias = sum(dout, dim=0)
    Tensor& dbias = bias_.Grad();
    dbias.Zeros();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        dbias[j] += dout_2d[i * out_features_ + j];
      }
    }

    // dx_2d ya contiene el resultado; basta devolverlo con el rango original.
    dx_2d.Reshape(last_input_.Shape());
    return dx_2d;
  }

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
