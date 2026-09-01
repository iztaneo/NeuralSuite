// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/maxpool2d.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/maxpool2d.h"

namespace neuralsuite {

Tensor MaxPool2D::Forward(const Tensor& input) {
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

    // Reparto por plano (muestra, canal): cada uno escribe en su propia franja
    // de la salida, sin reduccion, asi que el resultado es identico bit a bit
    // al de un solo hilo.
    parallel::ParallelFor(batch_size * channels, /*min_per_thread=*/1,
                          [&](int desde, int hasta) {
    for (int plano = desde; plano < hasta; ++plano) {
      const int b = plano / channels, c = plano % channels;
      {
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
    });
    return output;
  }

Tensor MaxPool2D::Backward(const Tensor& dout) {
    Tensor dx(last_input_.Shape());
    dx.Zeros();

    const int batch_size = last_input_.Shape()[0];
    const int channels = last_input_.Shape()[1];
    const int por_plano = static_cast<int>(dout.TotalSize()) / (batch_size * channels);

    parallel::ParallelFor(batch_size * channels, /*min_per_thread=*/1,
                          [&](int desde, int hasta) {
      for (int plano = desde; plano < hasta; ++plano) {
        const size_t inicio = static_cast<size_t>(plano) * por_plano;
        for (int k = 0; k < por_plano; ++k) {
          dx[max_indices_[inicio + k]] += dout[inicio + k];
        }
      }
    });
    return dx;
  }

}  // namespace neuralsuite
