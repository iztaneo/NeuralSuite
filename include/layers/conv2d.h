// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file conv2d.h
 * @brief 2D Convolutional Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parallel.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class Conv2DReference
 * @brief Convolucion escrita como su definicion: siete bucles anidados.
 *
 * Se conserva a proposito. Es lenta —1.5 GFLOP/s frente a los 232 que alcanza
 * `MatMul`— pero se lee al lado de la formula y no tiene nada que pueda
 * desalinearse: ni reordenaciones de memoria, ni indices calculados. `Conv2D`,
 * que si los tiene, se comprueba contra esta.
 *
 * Que el oraculo viva en el mismo archivo y se compile siempre es deliberado.
 * Una version de referencia que haya que desenterrar de un commit antiguo deja
 * de usarse a la primera prisa.
 */
class Conv2DReference : public Layer {
 public:
  Conv2DReference(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
      : in_channels_(in_ch),
        out_channels_(out_ch),
        kernel_size_(k_size),
        stride_(str),
        padding_(pad),
        weight_({out_ch, in_ch, k_size, k_size}),
        bias_({out_ch}) {
    Register(&weight_, "weight");
    Register(&bias_, "bias");
    weight_.Value().XavierInit(in_ch * k_size * k_size, out_ch * k_size * k_size);
    bias_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

 private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  Parameter weight_;
  Parameter bias_;
  Tensor last_input_;
};


/**
 * @class Conv2D
 * @brief Convolucion resuelta como una multiplicacion de matrices.
 *
 * La definicion de la convolucion son siete bucles anidados, y escrita asi
 * alcanzaba 1.5 GFLOP/s. Perfilando un paso de entrenamiento del CRNN de OCR,
 * las tres convoluciones se llevaban el 85.7% del tiempo: 514 ms la de 16 a 32
 * canales, 410 ms la de 32 a 64. Y corrian en un solo hilo teniendo diez, sin
 * tocar el pool que ya existia.
 *
 * La reformulacion es la clasica, `im2col`: cada ventana de la imagen se copia
 * como una columna de una matriz, y entonces la convolucion entera es
 *
 *     salida[canal, posicion] = pesos[canal, k] * columnas[k, posicion]
 *
 * con `k = canales_entrada * tamano * tamano`. Eso es exactamente un `MatMul`,
 * que en este proyecto ya esta paralelizado y con bloqueo de registros. Se
 * cambia trabajo por memoria: las columnas repiten cada pixel tantas veces como
 * ventanas lo cubren, `k` veces en el peor caso.
 *
 * El orden de los ejes no es casual. Las columnas se construyen ya
 * transpuestas, `[k, posiciones]`, para que los pesos entren en la
 * multiplicacion con la forma en que estan guardados —`[canales, k]`— y la
 * salida salga directamente como `[canales, posiciones]`, que es la disposicion
 * del tensor de salida. Cualquier otro orden obliga a transponer el resultado
 * de cada lote.
 *
 * No da los mismos bits que `Conv2DReference`: la suma se hace en otro orden y
 * en punto flotante eso importa. Lo que si debe cumplir es coincidir dentro del
 * redondeo, y de eso se encarga la prueba que las contrasta.
 *
 * El reparto entre hilos se hace **por el lote**, no dentro de la
 * multiplicacion. `MatMul` reparte por filas del resultado, y en una
 * convolucion esas filas son los canales de salida: 16, 32 o 64. Con el minimo
 * de trabajo por hilo que exige `ParallelFor`, eso dejaba la primera
 * convolucion en un solo hilo y la segunda en dos, teniendo diez. Las imagenes
 * de un lote, en cambio, son trabajos independientes y hay tantas como haga
 * falta.
 *
 * En el paso hacia atras los gradientes de los pesos suman sobre todo el lote,
 * que es una reduccion. Para que el resultado no dependa de cuantos hilos haya,
 * cada imagen escribe su contribucion en su propio hueco y la suma se hace
 * despues en orden fijo. Cuesta memoria y a cambio el gradiente es el mismo con
 * uno o con diez hilos.
 */
class Conv2D : public Layer {
 public:
  Conv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
      : in_channels_(in_ch),
        out_channels_(out_ch),
        kernel_size_(k_size),
        stride_(str),
        padding_(pad),
        weight_({out_ch, in_ch, k_size, k_size}),
        bias_({out_ch}) {
    if (in_ch <= 0 || out_ch <= 0 || k_size <= 0 || str <= 0 || pad < 0) {
      throw std::invalid_argument("Conv2D: parametros de la convolucion no validos.");
    }
    Register(&weight_, "weight");
    Register(&bias_, "bias");
    weight_.Value().XavierInit(in_ch * k_size * k_size, out_ch * k_size * k_size);
    bias_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Tensor& Weight() const { return weight_.Value(); }
  Tensor& Weight() { return weight_.Value(); }
  [[nodiscard]] const Tensor& Bias() const { return bias_.Value(); }
  Tensor& Bias() { return bias_.Value(); }

 private:
  /**
   * @brief Copia cada ventana de la imagen `b` como una columna.
   *
   * `salida` queda como `[canales*tamano*tamano, posiciones]`. Los pixeles que
   * caen fuera por el relleno se dejan a cero, que es lo que significa rellenar.
   */
  void Im2Col(const Tensor& entrada, int b, int height, int width, int out_h, int out_w, Tensor* salida) const;

  /** @brief La inversa de Im2Col: devuelve cada columna a su sitio, sumando. */
  void Col2Im(const Tensor& columnas, int b, int height, int width, int out_h, int out_w, Tensor* destino) const;

  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  Parameter weight_;
  Parameter bias_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
