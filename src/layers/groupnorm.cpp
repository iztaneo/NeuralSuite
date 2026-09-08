// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/groupnorm.h.

#include "layers/groupnorm.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel.h"

namespace neuralsuite {

namespace {

// Descompone [N, C, ...] en ejemplos, canales y tamano espacial.
void Reparto(const Tensor& t, int canales_esperados, int* n, int* c, int* espacial) {
  const std::vector<int>& forma = t.Shape();
  if (forma.size() < 2) {
    throw std::invalid_argument("GroupNorm: la entrada debe ser [N, C, ...].");
  }
  *n = forma[0];
  *c = forma[1];
  if (*c != canales_esperados) {
    throw std::invalid_argument(
        "GroupNorm: la entrada trae " + std::to_string(*c) + " canales y la capa "
        "se creo para " + std::to_string(canales_esperados) + ".");
  }
  *espacial = 1;
  for (size_t d = 2; d < forma.size(); ++d) *espacial *= forma[d];
}

}  // namespace

Tensor GroupNormLayer::Forward(const Tensor& input) {
  last_input_ = input;
  int n = 0, c = 0, espacial = 0;
  Reparto(input, canales_, &n, &c, &espacial);

  const int cpg = canales_ / grupos_;          // canales por grupo
  const int m = cpg * espacial;                // elementos sobre los que se normaliza

  Tensor output(input.Shape());
  media_.Resize({n * grupos_});
  rstd_.Resize({n * grupos_});

  // Se reparte por (ejemplo, grupo): cada uno escribe su propio tramo de la
  // salida y su propia media, asi que no hay reduccion entre hilos.
  parallel::ParallelFor(n * grupos_, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int ng = desde; ng < hasta; ++ng) {
      const int ejemplo = ng / grupos_;
      const int grupo = ng % grupos_;
      const size_t base = (static_cast<size_t>(ejemplo) * canales_ + grupo * cpg) *
                          espacial;

      // En double: sumar m valores en float pierde precision justo antes de una
      // raiz, y `m` aqui es canales_por_grupo * alto * ancho, que crece rapido.
      double suma = 0.0, suma2 = 0.0;
      for (int i = 0; i < m; ++i) {
        const double v = input[base + i];
        suma += v;
        suma2 += v * v;
      }
      const double mu = suma / m;
      const double var = suma2 / m - mu * mu;
      const float inv = static_cast<float>(1.0 / std::sqrt(var + eps_));
      media_[ng] = static_cast<float>(mu);
      rstd_[ng] = inv;

      for (int k = 0; k < cpg; ++k) {
        const int canal = grupo * cpg + k;
        const float g = gamma_.Value()[canal];
        const float b = beta_.Value()[canal];
        const size_t off = base + static_cast<size_t>(k) * espacial;
        for (int s = 0; s < espacial; ++s) {
          const float xhat = (input[off + s] - static_cast<float>(mu)) * inv;
          output[off + s] = xhat * g + b;
        }
      }
    }
  });
  return output;
}

Tensor GroupNormLayer::Backward(const Tensor& dout) {
  int n = 0, c = 0, espacial = 0;
  Reparto(last_input_, canales_, &n, &c, &espacial);

  const int cpg = canales_ / grupos_;
  const int m = cpg * espacial;

  Tensor dx(last_input_.Shape());

  // 1. dx: se reparte por (ejemplo, grupo), que son tramos disjuntos.
  parallel::ParallelFor(n * grupos_, /*min_per_thread=*/2, [&](int desde, int hasta) {
    for (int ng = desde; ng < hasta; ++ng) {
      const int ejemplo = ng / grupos_;
      const int grupo = ng % grupos_;
      const size_t base = (static_cast<size_t>(ejemplo) * canales_ + grupo * cpg) *
                          espacial;
      const float mu = media_[ng];
      const float inv = rstd_[ng];

      // Dos sumas sobre el grupo: la de los gradientes y la de su producto con
      // la entrada normalizada. Son los dos terminos que restan, y omitir
      // cualquiera de ellos deja un gradiente que parece razonable y no lo es.
      double suma_g = 0.0, suma_gx = 0.0;
      for (int k = 0; k < cpg; ++k) {
        const float gam = gamma_.Value()[grupo * cpg + k];
        const size_t off = base + static_cast<size_t>(k) * espacial;
        for (int s = 0; s < espacial; ++s) {
          const double g = static_cast<double>(dout[off + s]) * gam;
          suma_g += g;
          suma_gx += g * (last_input_[off + s] - mu) * inv;
        }
      }

      for (int k = 0; k < cpg; ++k) {
        const float gam = gamma_.Value()[grupo * cpg + k];
        const size_t off = base + static_cast<size_t>(k) * espacial;
        for (int s = 0; s < espacial; ++s) {
          const float xhat = (last_input_[off + s] - mu) * inv;
          const float g = dout[off + s] * gam;
          dx[off + s] = inv * (g - static_cast<float>(suma_g / m) -
                               xhat * static_cast<float>(suma_gx / m));
        }
      }
    }
  });

  // 2. dgamma y dbeta: se reparten por CANAL. Cada uno suma sobre ejemplos y
  //    posiciones, asi que repartir por ejemplos obligaria a reducir entre
  //    hilos; por canal, cada hilo escribe posiciones distintas.
  Tensor& dgamma = gamma_.Grad();
  Tensor& dbeta = beta_.Grad();
  parallel::ParallelFor(canales_, /*min_per_thread=*/8, [&](int desde, int hasta) {
    for (int canal = desde; canal < hasta; ++canal) {
      const int grupo = canal / cpg;
      double acc_g = 0.0, acc_b = 0.0;
      for (int ejemplo = 0; ejemplo < n; ++ejemplo) {
        const int ng = ejemplo * grupos_ + grupo;
        const float mu = media_[ng];
        const float inv = rstd_[ng];
        const size_t off = (static_cast<size_t>(ejemplo) * canales_ + canal) * espacial;
        for (int s = 0; s < espacial; ++s) {
          const float d = dout[off + s];
          acc_g += static_cast<double>(d) * (last_input_[off + s] - mu) * inv;
          acc_b += d;
        }
      }
      dgamma[canal] = static_cast<float>(acc_g);
      dbeta[canal] = static_cast<float>(acc_b);
    }
  });

  return dx;
}

}  // namespace neuralsuite
