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

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int channels = input.Shape()[1];
    int height = input.Shape()[2];
    int width = input.Shape()[3];

    int out_h = (height - pool_h_) / stride_h_ + 1;
    int out_w = (width - pool_w_) / stride_w_ + 1;
    if (out_h <= 0 || out_w <= 0) {
      throw std::invalid_argument(
          "MaxPool2D: la ventana " + std::to_string(pool_h_) + "x" + std::to_string(pool_w_) +
          " no cabe en una entrada de " + std::to_string(height) + "x" + std::to_string(width) + ".");
    }

    Tensor output({batch_size, channels, out_h, out_w});
    max_indices_.resize(output.TotalSize());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            float max_val = -std::numeric_limits<float>::infinity();
            size_t max_idx = 0;

            for (int ph = 0; ph < pool_h_; ++ph) {
              for (int pw = 0; pw < pool_w_; ++pw) {
                int ih = oh * stride_h_ + ph;
                int iw = ow * stride_w_ + pw;
                size_t in_idx = ((b * channels + c) * height + ih) * width + iw;
                if (input[in_idx] > max_val) {
                  max_val = input[in_idx];
                  max_idx = in_idx;
                }
              }
            }

            size_t out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[out_idx] = max_val;
            max_indices_[out_idx] = max_idx;
          }
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    Tensor dx(last_input_.Shape());
    dx.Zeros();

    size_t out_sz = dout.TotalSize();
    for (size_t i = 0; i < out_sz; ++i) {
      size_t max_idx = max_indices_[i];
      dx[max_idx] += dout[i];
    }
    return dx;
  }

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
