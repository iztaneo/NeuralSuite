// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de latent/gaussiana.h.

#include "latent/gaussiana.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace neuralsuite {
namespace latent {

Tensor GaussianaDiagonal::Forward(const Tensor& params, const Tensor& ruido) {
  const auto& f = params.Shape();
  if (f.size() != 4 || f[1] % 2 != 0) {
    throw std::invalid_argument(
        "GaussianaDiagonal: params debe ser [N, 2C, h, w] con un numero par de canales.");
  }
  n_ = f[0];
  c_ = f[1] / 2;
  espacial_ = static_cast<size_t>(f[2]) * static_cast<size_t>(f[3]);
  const std::vector<int> forma_lat = {n_, c_, f[2], f[3]};
  if (ruido.Shape() != forma_lat) {
    throw std::invalid_argument("GaussianaDiagonal: el ruido debe tener la forma del latente [N, " +
                                std::to_string(c_) + ", h, w].");
  }

  mu_ = Tensor(forma_lat);
  logvar_ = Tensor(forma_lat);
  sigma_ = Tensor(forma_lat);
  ruido_ = ruido;
  Tensor z(forma_lat);
  const size_t por_ejemplo = static_cast<size_t>(c_) * espacial_;
  recortado_.assign(mu_.TotalSize(), 0);

  for (int n = 0; n < n_; ++n) {
    // Los C primeros canales son la media; los C siguientes, la log-varianza.
    const size_t base_p = static_cast<size_t>(n) * 2 * por_ejemplo;
    const size_t base_l = static_cast<size_t>(n) * por_ejemplo;
    for (size_t i = 0; i < por_ejemplo; ++i) {
      const float m = params[base_p + i];
      const float lv_crudo = params[base_p + por_ejemplo + i];
      const float lv = std::min(kLogVarMax, std::max(kLogVarMin, lv_crudo));
      recortado_[base_l + i] = (lv != lv_crudo) ? 1 : 0;
      const float s = std::exp(0.5f * lv);
      mu_[base_l + i] = m;
      logvar_[base_l + i] = lv;
      sigma_[base_l + i] = s;
      z[base_l + i] = m + s * ruido[base_l + i];
    }
  }
  return z;
}

double GaussianaDiagonal::KL() const {
  if (n_ == 0) throw std::logic_error("GaussianaDiagonal: KL() antes de Forward().");
  double suma = 0.0;
  for (size_t i = 0; i < mu_.TotalSize(); ++i) {
    const double m = mu_[i], lv = logvar_[i];
    suma += 0.5 * (m * m + std::exp(lv) - 1.0 - lv);
  }
  return suma / n_;
}

Tensor GaussianaDiagonal::Backward(const Tensor& dz, float peso_kl) const {
  if (n_ == 0) throw std::logic_error("GaussianaDiagonal: Backward() antes de Forward().");
  if (dz.Shape() != mu_.Shape()) {
    throw std::invalid_argument("GaussianaDiagonal: dz debe tener la forma del latente.");
  }
  const size_t por_ejemplo = static_cast<size_t>(c_) * espacial_;
  Tensor dparams({n_, 2 * c_, mu_.Shape()[2], mu_.Shape()[3]});
  const double escala_kl = static_cast<double>(peso_kl) / n_;

  for (int n = 0; n < n_; ++n) {
    const size_t base_p = static_cast<size_t>(n) * 2 * por_ejemplo;
    const size_t base_l = static_cast<size_t>(n) * por_ejemplo;
    for (size_t i = 0; i < por_ejemplo; ++i) {
      const size_t j = base_l + i;
      // z = mu + sigma * ruido  ->  dz/dmu = 1
      // KL = 1/N sum 0.5 (mu^2 + e^lv - 1 - lv)  ->  dKL/dmu = mu / N
      dparams[base_p + i] = static_cast<float>(dz[j] + escala_kl * mu_[j]);

      // dz/dlv = ruido * sigma / 2, porque sigma = e^(lv/2)
      // dKL/dlv = 0.5 (e^lv - 1) / N
      // El factor 1/2 de la primera es el error tipico de este calculo.
      double d_lv = static_cast<double>(dz[j]) * ruido_[j] * 0.5 * sigma_[j] +
                    escala_kl * 0.5 * (std::exp(static_cast<double>(logvar_[j])) - 1.0);
      if (recortado_[j]) d_lv = 0.0;   // donde actua el recorte no pasa gradiente
      dparams[base_p + por_ejemplo + i] = static_cast<float>(d_lv);
    }
  }
  return dparams;
}

}  // namespace latent
}  // namespace neuralsuite
