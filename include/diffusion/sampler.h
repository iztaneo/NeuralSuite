// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

#ifndef NEURAL_SUITE_INCLUDE_DIFFUSION_SAMPLER_H_
#define NEURAL_SUITE_INCLUDE_DIFFUSION_SAMPLER_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "diffusion/schedule.h"
#include "tensor.h"

namespace neuralsuite {
namespace diffusion {

/**
 * @brief Lo unico que un muestreador necesita del modelo: `(x_t, t) -> eps`.
 *
 * Se toma como funcion y no como una `UNet2D` a proposito. El muestreador no
 * usa nada mas de la red, y desacoplarlo permite probarlo con un **oraculo
 * analitico** cuya respuesta exacta se conoce, sin modelo entrenado de por
 * medio. Sin eso, la unica forma de saber si el muestreador esta bien seria
 * mirar si las imagenes salen bonitas, que no distingue un muestreador correcto
 * con un modelo mediocre de uno roto con un modelo bueno.
 */
using Predictor = std::function<Tensor(const Tensor& x_t, const Tensor& t)>;

/**
 * @brief Fuente del ruido que se suma en cada paso estocastico.
 *
 * Poder inyectarla es lo que hace el muestreo reproducible y comparable contra
 * PyTorch: si el ruido lo generase el muestreador por su cuenta, dos
 * implementaciones correctas darian trayectorias distintas y la paridad no
 * podria distinguirlas de dos incorrectas.
 */
using FuenteRuido = std::function<void(int paso, Tensor* destino)>;

/** @brief Fuente por defecto: gaussiana con la semilla dada. */
FuenteRuido RuidoGaussiano(uint32_t semilla);

/** @brief Fuente que devuelve siempre cero; convierte DDPM en determinista. */
FuenteRuido RuidoNulo();

/**
 * @brief Muestreador DDPM: recorre los `T` pasos del calendario, uno a uno.
 *
 * En cada paso deshace un poco de ruido y vuelve a anadir un poco menos:
 *
 *     media   = (x_t - beta[t]/sqrt(1-ab[t]) * eps) / sqrt(alpha[t])
 *     x_{t-1} = media + sqrt(var[t]) * z
 *
 * con `var[t] = beta[t] * (1 - ab[t-1]) / (1 - ab[t])` —la varianza posterior,
 * no `beta[t]` a secas— y sin ruido en el ultimo paso. Es fiel y caro: `T`
 * evaluaciones de la red por imagen. `DDIMSampler` es la version barata.
 */
class DDPMSampler {
 public:
  explicit DDPMSampler(const DiffusionSchedule& calendario);

  /**
   * @brief Parte de `x_T` y devuelve `x_0`.
   *
   * `x_inicial` es `[N, C, H, W]`; normalmente ruido gaussiano, pero se recibe
   * hecho para poder repetir un muestreo exacto.
   */
  [[nodiscard]] Tensor Muestrear(const Predictor& modelo, const Tensor& x_inicial,
                                 const FuenteRuido& ruido) const;

  /** @brief Si `x_0` estimado se recorta a [-1, 1] en cada paso. */
  void RecortarX0(bool activo) { recortar_ = activo; }

 private:
  const DiffusionSchedule& cal_;
  bool recortar_ = false;
};

/**
 * @brief Muestreador DDIM: el mismo modelo, muchos menos pasos.
 *
 * DDPM necesita los `T` pasos porque su cadena es markoviana: cada `x_{t-1}`
 * depende de `x_t` y de nada mas, asi que no se pueden saltar eslabones. DDIM
 * reescribe el proceso de forma **no markoviana** con las mismas marginales, y
 * eso permite recorrer una subsecuencia: 50 pasos en vez de 1000.
 *
 *     x0_est  = (x - sqrt(1-ab[t]) * eps) / sqrt(ab[t])
 *     sigma   = eta * sqrt((1-ab[p])/(1-ab[t])) * sqrt(1 - ab[t]/ab[p])
 *     x_p     = sqrt(ab[p]) * x0_est + sqrt(1 - ab[p] - sigma^2) * eps + sigma*z
 *
 * Con `eta = 0` es determinista —misma semilla inicial, misma imagen— y con
 * `eta = 1` recupera la varianza de DDPM. Ese caso limite es comprobable y la
 * prueba unitaria lo comprueba: sobre la secuencia completa, `sigma` tiene que
 * coincidir con la desviacion posterior de DDPM.
 */
class DDIMSampler {
 public:
  /**
   * @param n_pasos cuantos pasos recorrer; `<= calendario.Pasos()`.
   * @param eta 0 determinista, 1 equivale a DDPM.
   */
  DDIMSampler(const DiffusionSchedule& calendario, int n_pasos, float eta = 0.0f);

  [[nodiscard]] Tensor Muestrear(const Predictor& modelo, const Tensor& x_inicial,
                                 const FuenteRuido& ruido) const;

  /** @brief La subsecuencia de pasos, de mayor a menor. */
  [[nodiscard]] const std::vector<int>& Subsecuencia() const { return taus_; }

  void RecortarX0(bool activo) { recortar_ = activo; }

 private:
  const DiffusionSchedule& cal_;
  std::vector<int> taus_;
  float eta_;
  bool recortar_ = false;
};

}  // namespace diffusion
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DIFFUSION_SAMPLER_H_
