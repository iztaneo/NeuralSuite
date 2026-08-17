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

namespace neuralsuite {

/**
 * @class Linear
 * @brief Dense Linear Layer Y = X * W + b for N-Dimensional Tensors.
 */
class Linear : public Layer {
 public:
  Linear(int in_features, int out_features)
      : in_features_(in_features),
        out_features_(out_features),
        weight_({in_features, out_features}),
        bias_({out_features}),
        dweight_({in_features, out_features}),
        dbias_({out_features}) {
    weight_.RandomNormal(0.0f, 0.02f);
    bias_.Zeros();
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
    MatMul(input_2d, weight_, output_2d);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        output_2d[i * out_features_ + j] += bias_[j];
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
    Tensor weight_t = Transpose(weight_);
    MatMul(dout_2d, weight_t, dx_2d);

    // dweight = input_2d^T * dout_2d ([in, N] * [N, out] -> [in, out])
    Tensor input_t = Transpose(input_2d);
    MatMul(input_t, dout_2d, dweight_);

    // dbias = sum(dout, dim=0)
    dbias_.Zeros();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < out_features_; ++j) {
        dbias_[j] += dout_2d[i * out_features_ + j];
      }
    }

    // dx_2d ya contiene el resultado; basta devolverlo con el rango original.
    dx_2d.Reshape(last_input_.Shape());
    return dx_2d;
  }

  std::vector<Tensor*> GetParameters() override { return {&weight_, &bias_}; }
  std::vector<Tensor*> GetGradients() override { return {&dweight_, &dbias_}; }

  [[nodiscard]] const Tensor& Weight() const { return weight_; }
  Tensor& Weight() { return weight_; }
  [[nodiscard]] const Tensor& Bias() const { return bias_; }
  Tensor& Bias() { return bias_; }

 private:
  int in_features_;
  int out_features_;

  Tensor weight_;
  Tensor bias_;
  Tensor dweight_;
  Tensor dbias_;

  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LINEAR_H_
