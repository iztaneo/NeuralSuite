// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file lstm.h
 * @brief LSTM Recurrent Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class LSTM
 * @brief Long Short-Term Memory Recurrent Cell Layer.
 */
class LSTM : public Layer {
 public:
  LSTM(int in_sz, int hid_sz)
      : input_size_(in_sz),
        hidden_size_(hid_sz),
        weight_ih_({4 * hid_sz, in_sz}),
        weight_hh_({4 * hid_sz, hid_sz}),
        bias_ih_({4 * hid_sz}),
        bias_hh_({4 * hid_sz}),
        dweight_ih_({4 * hid_sz, in_sz}),
        dweight_hh_({4 * hid_sz, hid_sz}),
        dbias_ih_({4 * hid_sz}),
        dbias_hh_({4 * hid_sz}) {
    weight_ih_.XavierInit(in_sz, 4 * hid_sz);
    weight_hh_.XavierInit(hid_sz, 4 * hid_sz);
    bias_ih_.Zeros();
    bias_hh_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int seq_len = input.Shape()[0];
    int batch_size = input.Shape()[1];

    Tensor output({seq_len, batch_size, hidden_size_});
    Tensor h({batch_size, hidden_size_}); h.Zeros();
    Tensor c({batch_size, hidden_size_}); c.Zeros();

    for (int t = 0; t < seq_len; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        for (int h_i = 0; h_i < hidden_size_; ++h_i) {
          size_t in_idx = (t * batch_size + b) * input_size_;
          float val = std::tanh(input[in_idx] + h[b * hidden_size_ + h_i]);
          h[b * hidden_size_ + h_i] = val;
          size_t out_idx = (t * batch_size + b) * hidden_size_ + h_i;
          output[out_idx] = val;
        }
      }
    }
    return output;
  }

  /**
   * @brief SIN IMPLEMENTAR: no calcula gradientes.
   *
   * @warning Este método es un marcador de posición, no un backward real. No
   * usa `dout`, deja los cuatro gradientes en cero y devuelve un dx nulo. En
   * consecuencia **la capa LSTM no aprende**: sus pesos nunca se actualizan, y
   * además corta la propagación hacia cualquier capa anterior.
   *
   * Un modelo que combine LSTM con otras capas seguirá reduciendo la pérdida
   * gracias a esas otras capas, lo que puede dar la falsa impresión de que el
   * LSTM está entrenando.
   *
   * Implementarlo requiere retropropagación a través del tiempo (BPTT) sobre
   * las cuatro puertas, con su correspondiente verificación por diferencias
   * finitas antes de considerarse correcto.
   */
  Tensor Backward(const Tensor& dout) override {
    (void)dout;  // sin usar: ver la advertencia de arriba
    Tensor dx(last_input_.Shape());
    dx.Zeros();
    dweight_ih_.Zeros();
    dweight_hh_.Zeros();
    dbias_ih_.Zeros();
    dbias_hh_.Zeros();
    return dx;
  }

  std::vector<Tensor*> GetParameters() override {
    return {&weight_ih_, &weight_hh_, &bias_ih_, &bias_hh_};
  }

  std::vector<Tensor*> GetGradients() override {
    return {&dweight_ih_, &dweight_hh_, &dbias_ih_, &dbias_hh_};
  }

 private:
  int input_size_;
  int hidden_size_;

  Tensor weight_ih_;
  Tensor weight_hh_;
  Tensor bias_ih_;
  Tensor bias_hh_;

  Tensor dweight_ih_;
  Tensor dweight_hh_;
  Tensor dbias_ih_;
  Tensor dbias_hh_;

  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
