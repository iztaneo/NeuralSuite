// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file maxpool2d.h
 * @brief 2D Max Pooling Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parallel.h"

namespace neuralsuite {

/**
 * @class MaxPool2D
 * @brief 2D Max Pooling Layer for spatial downsampling.
 *
 * La ventana puede no ser cuadrada. El CRNN de OCR necesita colapsar el alto
 * entero dejando el ancho intacto —una ventana `8x1`—, porque cada columna de
 * la imagen es un paso de la secuencia que luego lee el `BiLSTM`: reducir el
 * ancho perderia posiciones de texto.
 */
class MaxPool2D : public Layer {
 public:
  /** @brief Ventana cuadrada, el caso habitual. */
  explicit MaxPool2D(int p_size = 2, int str = 2)
      : MaxPool2D(p_size, p_size, str, str) {}

  /** @brief Ventana rectangular, con paso independiente en cada eje. */
  MaxPool2D(int pool_h, int pool_w, int stride_h, int stride_w)
      : pool_h_(pool_h), pool_w_(pool_w), stride_h_(stride_h), stride_w_(stride_w) {
    if (pool_h <= 0 || pool_w <= 0 || stride_h <= 0 || stride_w <= 0) {
      throw std::invalid_argument("MaxPool2D: ventana y paso deben ser positivos.");
    }
  }

  Tensor Forward(const Tensor& input) override;

  /**
   * @brief Devuelve el gradiente al maximo de cada ventana.
   *
   * El reparto va por plano y no por posicion de salida, y la diferencia
   * importa. Dos ventanas que se solapan —lo que ocurre cuando el paso es menor
   * que la ventana— pueden tener el mismo maximo, de modo que dos posiciones de
   * salida suman sobre la misma de entrada: repartir por posicion seria una
   * carrera. Por plano no puede haberla, porque planos distintos nunca
   * comparten posiciones de entrada, y ademas la suma dentro de cada plano
   * conserva su orden.
   */
  Tensor Backward(const Tensor& dout) override;

 private:
  int pool_h_;
  int pool_w_;
  int stride_h_;
  int stride_w_;
  Tensor last_input_;
  std::vector<size_t> max_indices_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_
