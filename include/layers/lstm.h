// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file lstm.h
 * @brief LSTM Recurrent Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_

#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parameter.h"
#include "../tensor.h"

namespace neuralsuite {

/**
 * @class LSTMReference
 * @brief La celda LSTM escrita como su definicion, unidad por unidad.
 *
 * Se conserva a proposito, igual que `Conv2DReference`: es lenta pero se lee al
 * lado de las ecuaciones y no tiene ninguna reordenacion de memoria que pueda
 * desalinearse. `LSTM`, que reformula lo mismo como multiplicaciones de
 * matrices, se comprueba contra esta.
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
class LSTMReference : public Layer {
 public:
  LSTMReference(int in_sz, int hid_sz)
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
      throw std::invalid_argument("LSTMReference: input_size y hidden_size deben ser positivos.");
    }
    weight_ih_.Value().XavierInit(in_sz, 4 * hid_sz);
    weight_hh_.Value().XavierInit(hid_sz, 4 * hid_sz);
    bias_ih_.Value().Zeros();
    bias_hh_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override;

  /**
   * @brief Retropropagación a través del tiempo (BPTT).
   *
   * Recorre la secuencia en sentido inverso acumulando en cada paso el
   * gradiente que llega desde la salida (`dout`) y el que arrastra el paso
   * siguiente a través de `h` y de `c`. Los gradientes de los parámetros se
   * acumulan sobre todos los pasos y todo el lote.
   */
  Tensor Backward(const Tensor& dout) override;

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
 * @class LSTM
 * @brief La misma celda, reformulada como multiplicaciones de matrices.
 *
 * `LSTMReference` calcula la preactivacion de cada puerta con un producto
 * escalar a mano: para cada paso de tiempo, cada muestra del lote, cada unidad
 * oculta y cada una de las cuatro puertas. Medido sobre el CRNN de OCR eso
 * daba 1.27 GFLOP/s, cuando `MatMul` en la misma maquina alcanza 232. Es el
 * mismo sintoma que tenia la convolucion antes de reformularla.
 *
 * La preactivacion es
 *
 *     pre_t = x_t · W_ih^T  +  h_{t-1} · W_hh^T  +  sesgos
 *
 * o sea dos multiplicaciones de matrices que estaban escritas como bucles. Al
 * reagruparlas, el trabajo se parte en dos mitades de naturaleza distinta:
 *
 *  - **La proyeccion de la entrada no depende del estado anterior**, asi que
 *    los `T` pasos se calculan de una vez, en una sola multiplicacion grande.
 *  - **La parte recurrente si depende**, y hay que recorrerla en orden. Pero
 *    cada paso pasa de `B · H · 4` productos escalares a una multiplicacion
 *    sobre todo el lote.
 *
 * Esa segunda mitad es el suelo: `h_t` necesita `h_{t-1}` y no hay forma de
 * solapar los pasos. Son matrices pequenas y ningun GEMM rinde al maximo con
 * ellas.
 *
 * En el paso hacia atras la asimetria se acentua. Solo el calculo de `dpre`
 * es secuencial; con todos los `dpre` juntos, los tres gradientes que quedan
 * salen de tres multiplicaciones grandes:
 *
 *     dW_ih = dpre^T · X        dW_hh = dpre^T · H_prev        dX = dpre · W_ih
 *
 * El truco esta en que `dW_hh` suma sobre todos los pasos y todo el lote, y
 * apilar las matrices de cada paso convierte esa suma en un unico producto.
 *
 * No da los mismos bits que `LSTMReference`: las sumas van en otro orden. Debe
 * coincidir dentro del redondeo, y de eso se encarga la prueba que las
 * contrasta, ademas de la paridad contra `nn.LSTM` de PyTorch.
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

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

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
  Tensor pre_;               // proyeccion de la entrada, [T*B, 4H]
  Tensor dpre_;              // gradiente de la preactivacion, [T*B, 4H]
  Tensor h_prev_apilado_;    // el estado que entra en cada paso, [T*B, H]

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
  Tensor Forward(const Tensor& input) override;

  /**
   * @brief `[T, B, 2H]` -> `[T, B, input_size]`.
   *
   * Cada mitad del gradiente vuelve por su celda. La entrada alimenta a las dos,
   * asi que las dos contribuciones se suman: es el mismo caso de un tensor que
   * se bifurca hacia dos consumidores.
   */
  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] int OutputSize() const { return 2 * hidden_size_; }

 private:
  /** @brief Invierte el eje temporal de un `[T, B, C]` dejando intacto el resto. */
  static Tensor ReverseTime(const Tensor& x);

  int input_size_;
  int hidden_size_;
  int seq_len_ = 0;
  int batch_size_ = 0;

  LSTM forward_;
  LSTM reverse_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_LSTM_H_
