// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file lstm.h
 * @brief LSTM Recurrent Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parameter.h"

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
        bias_hh_({4 * hid_sz}) {
    Register(&weight_ih_, "weight_ih");
    Register(&weight_hh_, "weight_hh");
    Register(&bias_ih_, "bias_ih");
    Register(&bias_hh_, "bias_hh");
    if (in_sz <= 0 || hid_sz <= 0) {
      throw std::invalid_argument("LSTM: input_size y hidden_size deben ser positivos.");
    }
    weight_ih_.Value().XavierInit(in_sz, 4 * hid_sz);
    weight_hh_.Value().XavierInit(hid_sz, 4 * hid_sz);
    bias_ih_.Value().Zeros();
    bias_hh_.Value().Zeros();
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
            float acc = bias_ih_.Value()[row] + bias_hh_.Value()[row];
            for (int j = 0; j < input_size_; ++j) {
              acc += weight_ih_.Value()[static_cast<size_t>(row) * input_size_ + j] * input[x_off + j];
            }
            if (t > 0) {
              for (int j = 0; j < H; ++j) {
                acc += weight_hh_.Value()[static_cast<size_t>(row) * H + j] * h_cache_[prev_off + j];
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

    Tensor& dweight_ih = weight_ih_.Grad();
    Tensor& dweight_hh = weight_hh_.Grad();
    Tensor& dbias_ih = bias_ih_.Grad();
    Tensor& dbias_hh = bias_hh_.Grad();
    dweight_ih.Zeros();
    dweight_hh.Zeros();
    dbias_ih.Zeros();
    dbias_hh.Zeros();

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

            dbias_ih[row] += d;
            dbias_hh[row] += d;

            for (int j = 0; j < input_size_; ++j) {
              dweight_ih[static_cast<size_t>(row) * input_size_ + j] += d * last_input_[x_off + j];
              dx[x_off + j] += d * weight_ih_.Value()[static_cast<size_t>(row) * input_size_ + j];
            }

            if (t > 0) {
              for (int j = 0; j < H; ++j) {
                dweight_hh[static_cast<size_t>(row) * H + j] += d * h_cache_[prev_off + j];
                dh_prev_acc[st_off + j] += d * weight_hh_.Value()[static_cast<size_t>(row) * H + j];
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

 private:
  static float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

  int input_size_;
  int hidden_size_;
  int seq_len_ = 0;
  int batch_size_ = 0;

  Parameter weight_ih_;
  Parameter weight_hh_;
  Parameter bias_ih_;
  Parameter bias_hh_;

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

/**
 * @class BiLSTM
 * @brief LSTM bidireccional: dos celdas independientes, una por sentido.
 *
 * La celda directa recorre la secuencia de `0` a `T-1` y la inversa de `T-1` a
 * `0`. La salida de cada paso es la concatenacion de ambos estados ocultos, de
 * modo que la forma pasa de `[T, B, H]` a `[T, B, 2H]`. Asi cada posicion ve
 * tanto lo que la precede como lo que la sigue, que es lo que necesita el OCR:
 * el trazo de una letra se interpreta mejor sabiendo con que continua.
 *
 * No reimplementa la recurrencia. Invierte el eje temporal de la entrada, se lo
 * pasa a una segunda `LSTM` ya verificada contra PyTorch y vuelve a invertir su
 * salida. La alternativa —escribir un bucle que recorra hacia atras— duplicaria
 * el BPTT, que es justo la parte donde es facil equivocarse.
 *
 * Corresponde a `nn.LSTM(..., bidirectional=True)` de PyTorch con una sola capa:
 * los parametros de la celda inversa son los que alli llevan el sufijo
 * `_reverse`, y la concatenacion de la salida sigue el mismo orden.
 */
class BiLSTM : public Layer {
 public:
  BiLSTM(int in_sz, int hid_sz)
      : input_size_(in_sz), hidden_size_(hid_sz), forward_(in_sz, hid_sz), reverse_(in_sz, hid_sz) {
    Register(&forward_, "forward");
    Register(&reverse_, "reverse");
  }

  /** @brief `[T, B, input_size]` -> `[T, B, 2 * hidden_size]`. */
  Tensor Forward(const Tensor& input) override {
    if (input.Shape().size() != 3) {
      throw std::invalid_argument("BiLSTM: la entrada debe ser [seq_len, batch, input_size].");
    }
    if (input.Shape()[2] != input_size_) {
      throw std::invalid_argument(
          "BiLSTM: la ultima dimension de la entrada es " + std::to_string(input.Shape()[2]) +
          " y se esperaba input_size = " + std::to_string(input_size_) + ".");
    }

    seq_len_ = input.Shape()[0];
    batch_size_ = input.Shape()[1];
    const int H = hidden_size_;

    const Tensor h_fwd = forward_.Forward(input);
    const Tensor h_rev = ReverseTime(reverse_.Forward(ReverseTime(input)));

    Tensor out({seq_len_, batch_size_, 2 * H});
    for (int t = 0; t < seq_len_; ++t) {
      for (int b = 0; b < batch_size_; ++b) {
        const size_t src = static_cast<size_t>(t * batch_size_ + b) * H;
        const size_t dst = static_cast<size_t>(t * batch_size_ + b) * 2 * H;
        for (int k = 0; k < H; ++k) {
          out[dst + k] = h_fwd[src + k];
          out[dst + H + k] = h_rev[src + k];
        }
      }
    }
    return out;
  }

  /**
   * @brief `[T, B, 2H]` -> `[T, B, input_size]`.
   *
   * Cada mitad del gradiente vuelve por su celda. La entrada alimenta a las dos,
   * asi que las dos contribuciones se suman: es el mismo caso de un tensor que
   * se bifurca hacia dos consumidores.
   */
  Tensor Backward(const Tensor& dout) override {
    const int H = hidden_size_;
    if (dout.TotalSize() != static_cast<size_t>(seq_len_) * batch_size_ * 2 * H) {
      throw std::invalid_argument("BiLSTM: dout no coincide con la forma de la salida.");
    }

    Tensor dh_fwd({seq_len_, batch_size_, H});
    Tensor dh_rev({seq_len_, batch_size_, H});
    for (int t = 0; t < seq_len_; ++t) {
      for (int b = 0; b < batch_size_; ++b) {
        const size_t src = static_cast<size_t>(t * batch_size_ + b) * 2 * H;
        const size_t dst = static_cast<size_t>(t * batch_size_ + b) * H;
        for (int k = 0; k < H; ++k) {
          dh_fwd[dst + k] = dout[src + k];
          dh_rev[dst + k] = dout[src + H + k];
        }
      }
    }

    const Tensor dx_fwd = forward_.Backward(dh_fwd);
    // La celda inversa vio la secuencia al reves: su gradiente entra invertido y
    // su resultado sale en ese mismo orden, de modo que hay que devolverlo.
    const Tensor dx_rev = ReverseTime(reverse_.Backward(ReverseTime(dh_rev)));

    Tensor dx({seq_len_, batch_size_, input_size_});
    for (size_t i = 0; i < dx.TotalSize(); ++i) dx[i] = dx_fwd[i] + dx_rev[i];
    return dx;
  }

  [[nodiscard]] int OutputSize() const { return 2 * hidden_size_; }

 private:
  /** @brief Invierte el eje temporal de un `[T, B, C]` dejando intacto el resto. */
  static Tensor ReverseTime(const Tensor& x) {
    const int T = x.Shape()[0];
    const int B = x.Shape()[1];
    const int C = x.Shape()[2];
    Tensor out({T, B, C});
    for (int t = 0; t < T; ++t) {
      const size_t src = static_cast<size_t>(t * B) * C;
      const size_t dst = static_cast<size_t>((T - 1 - t) * B) * C;
      std::memcpy(out.Data() + dst, x.Data() + src, static_cast<size_t>(B) * C * sizeof(float));
    }
    return out;
  }

  int input_size_;
  int hidden_size_;
  int seq_len_ = 0;
  int batch_size_ = 0;

  LSTM forward_;
  LSTM reverse_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
