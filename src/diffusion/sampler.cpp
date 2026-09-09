// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de diffusion/sampler.h.

#include "diffusion/sampler.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace neuralsuite {
namespace diffusion {

namespace {

/** @brief Generador propio, para no tocar el RNG global de los tensores. */
struct Xorshift {
  uint32_t s;
  float Normal() {
    // Box-Muller a partir de dos uniformes; basta para el ruido del muestreo.
    auto u = [&]() {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      return (static_cast<float>(s >> 8) + 0.5f) / 16777216.0f;  // (0, 1)
    };
    const float u1 = u(), u2 = u();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
  }
};

void Recortar(Tensor* x) {
  for (size_t i = 0; i < x->TotalSize(); ++i) {
    (*x)[i] = std::min(1.0f, std::max(-1.0f, (*x)[i]));
  }
}

/** @brief Comprueba que el predictor devolvio lo que se le pidio. */
void ComprobarPrediccion(const Tensor& eps, const Tensor& x, const char* quien) {
  if (eps.Shape() != x.Shape()) {
    throw std::runtime_error(std::string(quien) +
                             ": el predictor devolvio una forma distinta de la "
                             "de x_t; el modelo no encaja con el muestreo");
  }
}

}  // namespace

FuenteRuido RuidoGaussiano(uint32_t semilla) {
  // El estado va en un shared_ptr porque la fuente se copia al pasarla y el
  // ruido debe avanzar, no reiniciarse en cada paso.
  auto estado = std::make_shared<Xorshift>(Xorshift{semilla * 2654435761u + 1u});
  return [estado](int, Tensor* destino) {
    for (size_t i = 0; i < destino->TotalSize(); ++i) (*destino)[i] = estado->Normal();
  };
}

FuenteRuido RuidoNulo() {
  return [](int, Tensor* destino) { destino->Zeros(); };
}

// ---------------------------------------------------------------------------
// DDPM
// ---------------------------------------------------------------------------

DDPMSampler::DDPMSampler(const DiffusionSchedule& calendario) : cal_(calendario) {}

Tensor DDPMSampler::Muestrear(const Predictor& modelo, const Tensor& x_inicial,
                              const FuenteRuido& ruido) const {
  if (!modelo) throw std::invalid_argument("DDPMSampler: predictor vacio");
  if (!ruido) throw std::invalid_argument("DDPMSampler: fuente de ruido vacia");

  const int T = cal_.Pasos();
  const int N = x_inicial.Shape()[0];
  const Tensor& beta = cal_.Beta();
  const Tensor& alpha = cal_.Alpha();
  const Tensor& ab = cal_.AlphaBar();

  Tensor x = x_inicial;          // copia profunda: no se toca la entrada
  Tensor t({N}), z(x.Shape());

  for (int paso = T - 1; paso >= 0; --paso) {
    for (int n = 0; n < N; ++n) t[n] = static_cast<float>(paso);
    const Tensor eps = modelo(x, t);
    ComprobarPrediccion(eps, x, "DDPMSampler");

    // ab[-1] = 1 por convencion: antes del primer paso no se ha anadido ruido.
    const double ab_t = ab[paso];
    const double ab_prev = (paso > 0) ? static_cast<double>(ab[paso - 1]) : 1.0;
    const double b_t = beta[paso];
    const double inv_sqrt_a = 1.0 / std::sqrt(static_cast<double>(alpha[paso]));
    const double coef_eps = b_t / std::sqrt(1.0 - ab_t);
    // Varianza posterior, no beta[t] a secas. Las dos aparecen en el articulo
    // como cotas de la varianza real; esta es la que corresponde a condicionar
    // en x_0, y es la que usa la implementacion de referencia.
    const double var = b_t * (1.0 - ab_prev) / (1.0 - ab_t);
    const double desv = (paso > 0) ? std::sqrt(var) : 0.0;

    if (recortar_) {
      // Recortar actua sobre x_0 estimado, no sobre x_t: hay que despejarlo,
      // recortarlo y volver a componer la media a partir de el.
      Tensor x0(x.Shape());
      const double sq_ab = std::sqrt(ab_t), sq_um = std::sqrt(1.0 - ab_t);
      for (size_t i = 0; i < x.TotalSize(); ++i) {
        x0[i] = static_cast<float>((x[i] - sq_um * eps[i]) / sq_ab);
      }
      Recortar(&x0);
      const double c0 = std::sqrt(ab_prev) * b_t / (1.0 - ab_t);
      const double ct = std::sqrt(static_cast<double>(alpha[paso])) * (1.0 - ab_prev) / (1.0 - ab_t);
      for (size_t i = 0; i < x.TotalSize(); ++i) {
        x[i] = static_cast<float>(c0 * x0[i] + ct * x[i]);
      }
    } else {
      for (size_t i = 0; i < x.TotalSize(); ++i) {
        x[i] = static_cast<float>(inv_sqrt_a * (x[i] - coef_eps * eps[i]));
      }
    }

    if (desv > 0.0) {
      ruido(paso, &z);
      for (size_t i = 0; i < x.TotalSize(); ++i) {
        x[i] += static_cast<float>(desv * z[i]);
      }
    }
  }
  return x;
}

// ---------------------------------------------------------------------------
// DDIM
// ---------------------------------------------------------------------------

DDIMSampler::DDIMSampler(const DiffusionSchedule& calendario, int n_pasos, float eta)
    : cal_(calendario), eta_(eta) {
  const int T = calendario.Pasos();
  if (n_pasos <= 0 || n_pasos > T) {
    throw std::invalid_argument(
        "DDIMSampler: n_pasos debe estar entre 1 y " + std::to_string(T) +
        "; llego " + std::to_string(n_pasos));
  }
  if (eta < 0.0f || eta > 1.0f) {
    throw std::invalid_argument("DDIMSampler: eta debe estar en [0, 1]");
  }
  // Subsecuencia uniforme que **siempre incluye el paso T-1 y el 0**: empezar
  // por debajo de T-1 dejaria sin deshacer el ruido de los primeros pasos, y no
  // terminar en 0 devolveria un x_t, no un x_0.
  taus_.resize(n_pasos);
  for (int i = 0; i < n_pasos; ++i) {
    const double f = (n_pasos == 1) ? 0.0 : static_cast<double>(i) / (n_pasos - 1);
    taus_[i] = static_cast<int>(std::llround(f * (T - 1)));
  }
  std::reverse(taus_.begin(), taus_.end());   // de mayor a menor
}

Tensor DDIMSampler::Muestrear(const Predictor& modelo, const Tensor& x_inicial,
                              const FuenteRuido& ruido) const {
  if (!modelo) throw std::invalid_argument("DDIMSampler: predictor vacio");
  if (!ruido) throw std::invalid_argument("DDIMSampler: fuente de ruido vacia");

  const int N = x_inicial.Shape()[0];
  const Tensor& ab = cal_.AlphaBar();

  Tensor x = x_inicial;
  Tensor t({N}), z(x.Shape()), x0(x.Shape());

  for (size_t k = 0; k < taus_.size(); ++k) {
    const int tau = taus_[k];
    for (int n = 0; n < N; ++n) t[n] = static_cast<float>(tau);
    const Tensor eps = modelo(x, t);
    ComprobarPrediccion(eps, x, "DDIMSampler");

    const double ab_t = ab[tau];
    // El "anterior" es el siguiente de la subsecuencia, no tau-1: saltarse
    // eslabones es justo lo que DDIM permite. Tras el ultimo, ab = 1.
    const double ab_p = (k + 1 < taus_.size()) ? static_cast<double>(ab[taus_[k + 1]]) : 1.0;

    const double sq_ab = std::sqrt(ab_t), sq_um = std::sqrt(1.0 - ab_t);
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x0[i] = static_cast<float>((x[i] - sq_um * eps[i]) / sq_ab);
    }
    if (recortar_) Recortar(&x0);

    double sigma = 0.0;
    if (eta_ > 0.0f && ab_p < 1.0) {
      sigma = eta_ * std::sqrt((1.0 - ab_p) / (1.0 - ab_t)) *
              std::sqrt(1.0 - ab_t / ab_p);
    }
    // La direccion que apunta a x_t reutiliza el MISMO eps predicho; ese es el
    // truco de DDIM y la razon de que con eta=0 sea determinista.
    const double resto = 1.0 - ab_p - sigma * sigma;
    const double dir = std::sqrt(std::max(0.0, resto));
    const double sq_abp = std::sqrt(ab_p);

    if (sigma > 0.0) ruido(tau, &z);
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      double v = sq_abp * x0[i] + dir * eps[i];
      if (sigma > 0.0) v += sigma * z[i];
      x[i] = static_cast<float>(v);
    }
  }
  return x;
}

}  // namespace diffusion
}  // namespace neuralsuite
