// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file schedule.h
 * @brief El calendario de ruido de un modelo de difusion.
 */

#ifndef NEURAL_SUITE_INCLUDE_DIFFUSION_SCHEDULE_H_
#define NEURAL_SUITE_INCLUDE_DIFFUSION_SCHEDULE_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tensor.h"

namespace neuralsuite {
namespace diffusion {

/**
 * @class DiffusionSchedule
 * @brief Cuanto ruido lleva cada paso, y las cantidades que se derivan de eso.
 *
 * Un modelo de difusion aprende a deshacer un proceso conocido: partir de una
 * imagen y anadirle ruido gaussiano poco a poco hasta que solo queda ruido. El
 * calendario dice **cuanto** se anade en cada uno de los `T` pasos.
 *
 *     beta[t]       cuanta varianza se anade en el paso t
 *     alpha[t]      = 1 - beta[t], cuanto sobrevive de la senal
 *     alpha_bar[t]  producto acumulado de alpha hasta t
 *
 * `alpha_bar` es lo que hace practico entrenar: permite saltar directamente al
 * paso `t` sin recorrer los anteriores,
 *
 *     x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * ruido
 *
 * que es lo que hace `QSample`. Sin ese atajo habria que simular `t` pasos por
 * cada ejemplo del lote.
 *
 * La razon de que la suma sea `sqrt(ab) * x + sqrt(1-ab) * ruido` y no
 * `ab * x + (1-ab) * ruido` es que asi la **varianza se conserva**: si `x_0`
 * tiene varianza 1, `x_t` tambien, porque los coeficientes suman 1 al
 * cuadrado. Con la version sin raices la senal se apagaria mucho mas rapido de
 * lo que dice el calendario, y el modelo veria entradas de escala distinta a la
 * que espera. Es un error facil y silencioso.
 */
class DiffusionSchedule {
 public:
  /**
   * @brief Calendario lineal, el del articulo original de DDPM.
   *
   * `beta` crece linealmente de `beta_inicial` a `beta_final`. Los valores por
   * defecto —1e-4 y 0.02 con 1000 pasos— son los del articulo, y estan pensados
   * para que `alpha_bar` acabe cerca de cero: si no llega, el ultimo paso no es
   * ruido puro y el muestreo arranca de una distribucion que el modelo no vio.
   */
  explicit DiffusionSchedule(int pasos, float beta_inicial = 1e-4f,
                             float beta_final = 0.02f);

  [[nodiscard]] int Pasos() const { return pasos_; }
  [[nodiscard]] const Tensor& Beta() const { return beta_; }
  [[nodiscard]] const Tensor& Alpha() const { return alpha_; }
  [[nodiscard]] const Tensor& AlphaBar() const { return alpha_bar_; }

  /**
   * @brief `sqrt(ab[T-1])`: cuanta senal de `x_0` queda en `x_T`.
   *
   * Deberia ser practicamente cero. Si no lo es, el ultimo paso **no es ruido
   * puro**, el modelo nunca ve ruido puro durante el entrenamiento y el
   * muestreo arranca de una distribucion que no conoce: salen manchas, no
   * imagenes, y ni la perdida ni las diferencias finitas ni la paridad dicen
   * nada, porque las tres son correctas.
   *
   * Pasa con facilidad al acortar el calendario y dejar las betas del articulo,
   * que estan calibradas para 1000 pasos: con 200 pasos y `beta_final = 0.02`
   * queda el **36%** de la imagen. Por eso esto se expone y se comprueba.
   */
  [[nodiscard]] float SenalResidual() const {
    return std::sqrt(alpha_bar_[pasos_ - 1]);
  }

  /**
   * @brief `x_t = sqrt(ab[t]) * x_0 + sqrt(1 - ab[t]) * ruido`.
   *
   * `x0` y `ruido` son `[N, ...]`; `t` es `[N]` con el paso de cada ejemplo del
   * lote. Cada ejemplo lleva su propio `t` porque en entrenamiento se sortea uno
   * distinto por muestra: recorrer todos los pasos para cada imagen seria mil
   * veces mas caro y no aportaria nada.
   */
  void QSample(const Tensor& x0, const Tensor& ruido, const Tensor& t,
               Tensor* salida) const;

  /**
   * @brief Recupera `x_0` a partir de `x_t` y del ruido predicho.
   *
   *     x_0 = (x_t - sqrt(1 - ab[t]) * ruido) / sqrt(ab[t])
   *
   * Es despejar `QSample`. Se usa en el muestreo y para poder mirar que cree el
   * modelo que hay debajo del ruido, que es la forma mas directa de ver si esta
   * aprendiendo algo antes de que las imagenes finales sean reconocibles.
   */
  void PredecirX0(const Tensor& xt, const Tensor& ruido_predicho, const Tensor& t,
                  Tensor* salida) const;

 private:
  void Comprobar(const Tensor& x, const Tensor& t, const char* quien) const;

  int pasos_;
  Tensor beta_, alpha_, alpha_bar_;
  Tensor sqrt_ab_, sqrt_um_ab_;   // raices precalculadas; se usan en cada paso
};

}  // namespace diffusion
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DIFFUSION_SCHEDULE_H_
