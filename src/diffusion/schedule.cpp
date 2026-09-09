// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de diffusion/schedule.h.

#include "diffusion/schedule.h"

#include <cmath>

namespace neuralsuite {
namespace diffusion {

DiffusionSchedule::DiffusionSchedule(int pasos, float beta_inicial, float beta_final)
    : pasos_(pasos) {
  if (pasos <= 0) {
    throw std::invalid_argument("DiffusionSchedule: los pasos deben ser positivos.");
  }
  if (!(beta_inicial > 0.0f) || !(beta_final < 1.0f) || beta_inicial > beta_final) {
    // beta fuera de (0, 1) rompe el calendario en silencio: con beta >= 1 el
    // alpha correspondiente seria <= 0 y su raiz, NaN.
    throw std::invalid_argument(
        "DiffusionSchedule: beta debe cumplir 0 < inicial <= final < 1; se recibio " +
        std::to_string(beta_inicial) + " y " + std::to_string(beta_final) + ".");
  }

  beta_.Resize({pasos});
  alpha_.Resize({pasos});
  alpha_bar_.Resize({pasos});
  sqrt_ab_.Resize({pasos});
  sqrt_um_ab_.Resize({pasos});

  // El acumulado va en double por precaucion barata, no porque haga falta. Se
  // midio: con 1000 pasos, acumular en float se desvia 7.1e-07 en relativo y el
  // coeficiente del ruido cambia 1.5e-06, o sea nada. Una mutacion que lo pase a
  // float no la caza ninguna prueba, y esta bien que no la cace: no es un
  // defecto. Se deja en double porque no cuesta y porque con calendarios mas
  // largos la cuenta empeora, pero conviene no exagerar la razon.
  double acumulado = 1.0;
  for (int t = 0; t < pasos; ++t) {
    const float b = pasos == 1
                        ? beta_inicial
                        : beta_inicial + (beta_final - beta_inicial) *
                                             static_cast<float>(t) / (pasos - 1);
    beta_[t] = b;
    alpha_[t] = 1.0f - b;
    acumulado *= (1.0 - static_cast<double>(b));
    alpha_bar_[t] = static_cast<float>(acumulado);
    sqrt_ab_[t] = static_cast<float>(std::sqrt(acumulado));
    sqrt_um_ab_[t] = static_cast<float>(std::sqrt(1.0 - acumulado));
  }
}

void DiffusionSchedule::Comprobar(const Tensor& x, const Tensor& t,
                                  const char* quien) const {
  if (x.Shape().empty()) {
    throw std::invalid_argument(std::string(quien) + ": el tensor no tiene ejes.");
  }
  const int n = x.Shape()[0];
  if (static_cast<int>(t.TotalSize()) != n) {
    throw std::invalid_argument(
        std::string(quien) + ": hay " + std::to_string(n) + " ejemplos y " +
        std::to_string(t.TotalSize()) + " pasos; debe haber uno por ejemplo.");
  }
  for (size_t i = 0; i < t.TotalSize(); ++i) {
    const int paso = static_cast<int>(t[i]);
    if (paso < 0 || paso >= pasos_) {
      // Un paso fuera de rango leeria del calendario donde no debe y daria una
      // imagen con el ruido de otro momento, sin que nada fallara.
      throw std::out_of_range(std::string(quien) + ": el paso " + std::to_string(paso) +
                              " del ejemplo " + std::to_string(i) + " cae fuera de [0, " +
                              std::to_string(pasos_) + ").");
    }
  }
}

void DiffusionSchedule::QSample(const Tensor& x0, const Tensor& ruido, const Tensor& t,
                                Tensor* salida) const {
  Comprobar(x0, t, "QSample");
  if (ruido.Shape() != x0.Shape()) {
    throw std::invalid_argument("QSample: el ruido no tiene la forma de x0.");
  }

  const int n = x0.Shape()[0];
  const size_t por_ejemplo = x0.TotalSize() / static_cast<size_t>(n);
  salida->Resize(x0.Shape());

  for (int i = 0; i < n; ++i) {
    const int paso = static_cast<int>(t[i]);
    const float a = sqrt_ab_[paso];
    const float b = sqrt_um_ab_[paso];
    const size_t base = static_cast<size_t>(i) * por_ejemplo;
    for (size_t k = 0; k < por_ejemplo; ++k) {
      (*salida)[base + k] = a * x0[base + k] + b * ruido[base + k];
    }
  }
}

void DiffusionSchedule::PredecirX0(const Tensor& xt, const Tensor& ruido_predicho,
                                   const Tensor& t, Tensor* salida) const {
  Comprobar(xt, t, "PredecirX0");
  if (ruido_predicho.Shape() != xt.Shape()) {
    throw std::invalid_argument("PredecirX0: el ruido no tiene la forma de x_t.");
  }

  const int n = xt.Shape()[0];
  const size_t por_ejemplo = xt.TotalSize() / static_cast<size_t>(n);
  salida->Resize(xt.Shape());

  for (int i = 0; i < n; ++i) {
    const int paso = static_cast<int>(t[i]);
    const float a = sqrt_ab_[paso];
    const float b = sqrt_um_ab_[paso];
    const size_t base = static_cast<size_t>(i) * por_ejemplo;
    for (size_t k = 0; k < por_ejemplo; ++k) {
      (*salida)[base + k] = (xt[base + k] - b * ruido_predicho[base + k]) / a;
    }
  }
}

}  // namespace diffusion
}  // namespace neuralsuite
