// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file lstm.h
 * @brief LSTM Recurrent Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class LSTM
 * @brief Long Short-Term Memory Recurrent Cell Layer.
 *
 * Celda LSTM estándar sobre una secuencia de entrada `[seq_len, batch, input]`.
 * Para cada paso temporal se calcula la preactivación de las cuatro puertas
 *
 *     gates = W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh
 *
 * dividida en cuatro bloques de `hidden_size` en el orden `i, f, g, o`:
 *
 *     i = σ(gates_i)     puerta de entrada
 *     f = σ(gates_f)     puerta de olvido
 *     g = tanh(gates_g)  candidato de celda
 *     o = σ(gates_o)     puerta de salida
 *
 *     c_t = f ⊙ c_{t-1} + i ⊙ g
 *     h_t = o ⊙ tanh(c_t)
 *
 * El estado inicial `h_{-1}` y `c_{-1}` es cero. La salida es la secuencia
 * completa de estados ocultos, `[seq_len, batch, hidden]`.
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
    if (in_sz <= 0 || hid_sz <= 0) {
      throw std::invalid_argument("LSTM: input_size y hidden_size deben ser positivos.");
    }
    weight_ih_.XavierInit(in_sz, 4 * hid_sz);
    weight_hh_.XavierInit(hid_sz, 4 * hid_sz);
    bias_ih_.Zeros();
    bias_hh_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    if (input.Shape().size() != 3) {
      throw std::invalid_argument("LSTM: la entrada debe ser [seq_len, batch, input_size].");
    }
    if (input.Shape()[2] != input_size_) {
      throw std::invalid_argument(
          "LSTM: la ultima dimension de la entrada es " + std::to_string(input.Shape()[2]) +
          " y se esperaba input_size = " + std::to_string(input_size_) + ".");
    }

    last_input_ = input;
    const int seq_len = input.Shape()[0];
    const int batch_size = input.Shape()[1];
    const int H = hidden_size_;

    seq_len_ = seq_len;
    batch_size_ = batch_size;

    // Activaciones cacheadas para el backward. Se guarda tanh(c_t) porque el
    // gradiente de h respecto de c lo necesita y recomputarlo cuesta igual.
    const std::vector<int> gate_shape = {seq_len, batch_size, H};
    i_cache_.Resize(gate_shape);
    f_cache_.Resize(gate_shape);
    g_cache_.Resize(gate_shape);
    o_cache_.Resize(gate_shape);
    c_cache_.Resize(gate_shape);
    tanh_c_cache_.Resize(gate_shape);
    h_cache_.Resize(gate_shape);

    Tensor output({seq_len, batch_size, H});

    for (int t = 0; t < seq_len; ++t) {
      for (int b = 0; b < batch_size; ++b) {
        const size_t x_off = static_cast<size_t>(t * batch_size + b) * input_size_;
        const size_t s_off = static_cast<size_t>(t * batch_size + b) * H;
        const size_t prev_off = static_cast<size_t>((t - 1) * batch_size + b) * H;

        for (int k = 0; k < H; ++k) {
          // Preactivación de las cuatro puertas para la unidad k.
          float pre[4];
          for (int gate = 0; gate < 4; ++gate) {
            const int row = gate * H + k;
            float acc = bias_ih_[row] + bias_hh_[row];
            for (int j = 0; j < input_size_; ++j) {
              acc += weight_ih_[static_cast<size_t>(row) * input_size_ + j] * input[x_off + j];
            }
            if (t > 0) {
              for (int j = 0; j < H; ++j) {
                acc += weight_hh_[static_cast<size_t>(row) * H + j] * h_cache_[prev_off + j];
              }
            }
            pre[gate] = acc;
          }

          const float i_g = Sigmoid(pre[0]);
          const float f_g = Sigmoid(pre[1]);
          const float g_g = std::tanh(pre[2]);
          const float o_g = Sigmoid(pre[3]);

          const float c_prev = (t > 0) ? c_cache_[prev_off + k] : 0.0f;
          const float c_t = f_g * c_prev + i_g * g_g;
          const float tanh_c = std::tanh(c_t);
          const float h_t = o_g * tanh_c;

          i_cache_[s_off + k] = i_g;
          f_cache_[s_off + k] = f_g;
          g_cache_[s_off + k] = g_g;
          o_cache_[s_off + k] = o_g;
          c_cache_[s_off + k] = c_t;
          tanh_c_cache_[s_off + k] = tanh_c;
          h_cache_[s_off + k] = h_t;
          output[s_off + k] = h_t;
        }
      }
    }
    return output;
  }

  /**
   * @brief Retropropagación a través del tiempo (BPTT).
   *
   * Recorre la secuencia en sentido inverso acumulando en cada paso el
   * gradiente que llega desde la salida (`dout`) y el que arrastra el paso
   * siguiente a través de `h` y de `c`. Los gradientes de los parámetros se
   * acumulan sobre todos los pasos y todo el lote.
   */
  Tensor Backward(const Tensor& dout) override {
    const int seq_len = seq_len_;
    const int batch_size = batch_size_;
    const int H = hidden_size_;

    if (seq_len == 0) return Tensor();
    if (dout.TotalSize() != static_cast<size_t>(seq_len) * batch_size * H) {
      throw std::invalid_argument("LSTM: dout no coincide con la forma de la salida.");
    }

    dweight_ih_.Zeros();
    dweight_hh_.Zeros();
    dbias_ih_.Zeros();
    dbias_hh_.Zeros();

    Tensor dx(last_input_.Shape());
    dx.Zeros();

    // Gradiente que el paso t+1 propaga hacia t por el estado oculto y la celda.
    std::vector<float> dh_next(static_cast<size_t>(batch_size) * H, 0.0f);
    std::vector<float> dc_next(static_cast<size_t>(batch_size) * H, 0.0f);
    std::vector<float> dh_prev_acc(static_cast<size_t>(batch_size) * H, 0.0f);
    std::vector<float> dc_prev_acc(static_cast<size_t>(batch_size) * H, 0.0f);

    for (int t = seq_len - 1; t >= 0; --t) {
      std::fill(dh_prev_acc.begin(), dh_prev_acc.end(), 0.0f);
      std::fill(dc_prev_acc.begin(), dc_prev_acc.end(), 0.0f);

      for (int b = 0; b < batch_size; ++b) {
        const size_t x_off = static_cast<size_t>(t * batch_size + b) * input_size_;
        const size_t s_off = static_cast<size_t>(t * batch_size + b) * H;
        const size_t prev_off = static_cast<size_t>((t - 1) * batch_size + b) * H;
        const size_t st_off = static_cast<size_t>(b) * H;

        for (int k = 0; k < H; ++k) {
          const float i_g = i_cache_[s_off + k];
          const float f_g = f_cache_[s_off + k];
          const float g_g = g_cache_[s_off + k];
          const float o_g = o_cache_[s_off + k];
          const float tanh_c = tanh_c_cache_[s_off + k];
          const float c_prev = (t > 0) ? c_cache_[prev_off + k] : 0.0f;

          // El estado oculto influye en la pérdida por dos caminos: la salida
          // de este paso y el estado que recibe el paso siguiente.
          const float dh = dout[s_off + k] + dh_next[st_off + k];

          const float do_g = dh * tanh_c;
          const float dc = dh * o_g * (1.0f - tanh_c * tanh_c) + dc_next[st_off + k];

          const float di_g = dc * g_g;
          const float dg_g = dc * i_g;
          const float df_g = dc * c_prev;

          // Derivadas de las no linealidades de cada puerta.
          float dpre[4];
          dpre[0] = di_g * i_g * (1.0f - i_g);
          dpre[1] = df_g * f_g * (1.0f - f_g);
          dpre[2] = dg_g * (1.0f - g_g * g_g);
          dpre[3] = do_g * o_g * (1.0f - o_g);

          // La celda anterior recibe el gradiente atenuado por la puerta de olvido.
          dc_prev_acc[st_off + k] += dc * f_g;

          for (int gate = 0; gate < 4; ++gate) {
            const int row = gate * H + k;
            const float d = dpre[gate];

            dbias_ih_[row] += d;
            dbias_hh_[row] += d;

            for (int j = 0; j < input_size_; ++j) {
              dweight_ih_[static_cast<size_t>(row) * input_size_ + j] += d * last_input_[x_off + j];
              dx[x_off + j] += d * weight_ih_[static_cast<size_t>(row) * input_size_ + j];
            }

            if (t > 0) {
              for (int j = 0; j < H; ++j) {
                dweight_hh_[static_cast<size_t>(row) * H + j] += d * h_cache_[prev_off + j];
                dh_prev_acc[st_off + j] += d * weight_hh_[static_cast<size_t>(row) * H + j];
              }
            }
          }
        }
      }

      dh_next = dh_prev_acc;
      dc_next = dc_prev_acc;
    }

    return dx;
  }

  std::vector<Tensor*> GetParameters() override {
    return {&weight_ih_, &weight_hh_, &bias_ih_, &bias_hh_};
  }

  std::vector<Tensor*> GetGradients() override {
    return {&dweight_ih_, &dweight_hh_, &dbias_ih_, &dbias_hh_};
  }

 private:
  static float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

  int input_size_;
  int hidden_size_;
  int seq_len_ = 0;
  int batch_size_ = 0;

  Tensor weight_ih_;
  Tensor weight_hh_;
  Tensor bias_ih_;
  Tensor bias_hh_;

  Tensor dweight_ih_;
  Tensor dweight_hh_;
  Tensor dbias_ih_;
  Tensor dbias_hh_;

  Tensor last_input_;

  // Activaciones por paso temporal necesarias para el BPTT.
  Tensor i_cache_;
  Tensor f_cache_;
  Tensor g_cache_;
  Tensor o_cache_;
  Tensor c_cache_;
  Tensor tanh_c_cache_;
  Tensor h_cache_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
