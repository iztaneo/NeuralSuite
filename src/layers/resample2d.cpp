// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/resample2d.h.

#include "layers/resample2d.h"

#include "parallel.h"

namespace neuralsuite {

namespace {

// Comprueba [N, C, H, W] y descompone. Las dos capas trabajan sobre planos
// (ejemplo, canal) independientes, que es lo que permite repartirlas sin
// reduccion entre hilos.
void Reparto(const Tensor& t, const char* quien, int* n, int* c, int* h, int* w) {
  const std::vector<int>& f = t.Shape();
  if (f.size() != 4) {
    throw std::invalid_argument(std::string(quien) + ": la entrada debe ser " +
                                "[N, C, alto, ancho] y tiene " +
                                std::to_string(f.size()) + " ejes.");
  }
  *n = f[0]; *c = f[1]; *h = f[2]; *w = f[3];
}

}  // namespace

Tensor Upsample2D::Forward(const Tensor& input) {
  forma_entrada_ = input.Shape();
  int n = 0, c = 0, h = 0, w = 0;
  Reparto(input, "Upsample2D", &n, &c, &h, &w);

  const int hs = h * factor_, ws = w * factor_;
  Tensor output({n, c, hs, ws});

  parallel::ParallelFor(n * c, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int plano = desde; plano < hasta; ++plano) {
      const size_t base_in = static_cast<size_t>(plano) * h * w;
      const size_t base_out = static_cast<size_t>(plano) * hs * ws;
      for (int y = 0; y < hs; ++y) {
        const int sy = y / factor_;
        for (int x = 0; x < ws; ++x) {
          output[base_out + static_cast<size_t>(y) * ws + x] =
              input[base_in + static_cast<size_t>(sy) * w + x / factor_];
        }
      }
    }
  });
  return output;
}

Tensor Upsample2D::Backward(const Tensor& dout) {
  int n = forma_entrada_[0], c = forma_entrada_[1];
  const int h = forma_entrada_[2], w = forma_entrada_[3];
  const int hs = h * factor_, ws = w * factor_;

  Tensor dx(forma_entrada_);

  parallel::ParallelFor(n * c, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int plano = desde; plano < hasta; ++plano) {
      const size_t base_in = static_cast<size_t>(plano) * h * w;
      const size_t base_out = static_cast<size_t>(plano) * hs * ws;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          // Cada pixel de entrada se copio f*f veces: aqui se SUMAN los f*f
          // gradientes que le corresponden. Asignar en vez de sumar es el error
          // que deja el gradiente f*f veces mas pequeno.
          float acc = 0.0f;
          for (int dy = 0; dy < factor_; ++dy) {
            const size_t fila = base_out +
                                static_cast<size_t>(y * factor_ + dy) * ws +
                                static_cast<size_t>(x) * factor_;
            for (int dx_ = 0; dx_ < factor_; ++dx_) acc += dout[fila + dx_];
          }
          dx[base_in + static_cast<size_t>(y) * w + x] = acc;
        }
      }
    }
  });
  return dx;
}

Tensor Downsample2D::Forward(const Tensor& input) {
  forma_entrada_ = input.Shape();
  int n = 0, c = 0, h = 0, w = 0;
  Reparto(input, "Downsample2D", &n, &c, &h, &w);

  if (h % factor_ != 0 || w % factor_ != 0) {
    throw std::invalid_argument(
        "Downsample2D: " + std::to_string(h) + "x" + std::to_string(w) +
        " no es multiplo de " + std::to_string(factor_) +
        "; un borde sobrante habria que recortarlo o rellenarlo, y las dos "
        "opciones cambian el resultado en silencio.");
  }

  const int hs = h / factor_, ws = w / factor_;
  const float inv = 1.0f / static_cast<float>(factor_ * factor_);
  Tensor output({n, c, hs, ws});

  parallel::ParallelFor(n * c, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int plano = desde; plano < hasta; ++plano) {
      const size_t base_in = static_cast<size_t>(plano) * h * w;
      const size_t base_out = static_cast<size_t>(plano) * hs * ws;
      for (int y = 0; y < hs; ++y) {
        for (int x = 0; x < ws; ++x) {
          float acc = 0.0f;
          for (int dy = 0; dy < factor_; ++dy) {
            const size_t fila = base_in +
                                static_cast<size_t>(y * factor_ + dy) * w +
                                static_cast<size_t>(x) * factor_;
            for (int dx_ = 0; dx_ < factor_; ++dx_) acc += input[fila + dx_];
          }
          output[base_out + static_cast<size_t>(y) * ws + x] = acc * inv;
        }
      }
    }
  });
  return output;
}

Tensor Downsample2D::Backward(const Tensor& dout) {
  const int n = forma_entrada_[0], c = forma_entrada_[1];
  const int h = forma_entrada_[2], w = forma_entrada_[3];
  const int hs = h / factor_, ws = w / factor_;
  const float inv = 1.0f / static_cast<float>(factor_ * factor_);

  Tensor dx(forma_entrada_);

  parallel::ParallelFor(n * c, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int plano = desde; plano < hasta; ++plano) {
      const size_t base_in = static_cast<size_t>(plano) * h * w;
      const size_t base_out = static_cast<size_t>(plano) * hs * ws;
      for (int y = 0; y < hs; ++y) {
        for (int x = 0; x < ws; ++x) {
          // El promedio reparte: cada uno de los f*f recibe la misma fraccion.
          // Olvidar el 1/f² deja un gradiente f² veces mayor, que es la
          // simetria de lo que le pasa a Upsample2D si asigna en vez de sumar.
          const float parte = dout[base_out + static_cast<size_t>(y) * ws + x] * inv;
          for (int dy = 0; dy < factor_; ++dy) {
            const size_t fila = base_in +
                                static_cast<size_t>(y * factor_ + dy) * w +
                                static_cast<size_t>(x) * factor_;
            for (int dx_ = 0; dx_ < factor_; ++dx_) dx[fila + dx_] = parte;
          }
        }
      }
    }
  });
  return dx;
}

}  // namespace neuralsuite
