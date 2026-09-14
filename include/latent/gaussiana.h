// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file gaussiana.h
 * @brief Latente gaussiano del autoencoder: reparametrizacion y KL.
 */

#ifndef NEURAL_SUITE_INCLUDE_LATENT_GAUSSIANA_H_
#define NEURAL_SUITE_INCLUDE_LATENT_GAUSSIANA_H_

#include <cstdint>
#include <vector>

#include "../tensor.h"

namespace neuralsuite {
namespace latent {

/**
 * @class GaussianaDiagonal
 * @brief Convierte la salida del codificador en un latente muestreado.
 *
 * El codificador no devuelve un latente fijo sino los parametros de una
 * gaussiana por posicion: `[N, 2C, h, w]` se reparte en la media `mu` (los
 * primeros C canales) y el logaritmo de la varianza (los otros C). El latente se
 * sortea con el **truco de reparametrizacion**:
 *
 *     z = mu + exp(logvar / 2) * ruido,   ruido ~ N(0, 1)
 *
 * Escrito asi, el azar entra como una entrada mas y el gradiente atraviesa el
 * sorteo hasta `mu` y `logvar`. Sortear `z` directamente de N(mu, sigma) no
 * dejaria derivar nada.
 *
 * Y la **penalizacion KL** contra N(0, 1), por ejemplo y sumada sobre todo el
 * latente, promediada en el lote —la misma definicion que el codigo de Stable
 * Diffusion—:
 *
 *     KL = 1/N · sum 0.5 · (mu^2 + exp(logvar) - 1 - logvar)
 *
 * Sin ella el autoencoder puede llevar la varianza a cero y las medias a donde
 * quiera: reconstruye igual de bien, pero el latente queda desordenado y no se
 * puede difundir sobre el. Con media 0 y log-varianza 0 vale exactamente cero,
 * que es una comprobacion con respuesta conocida.
 *
 * `logvar` se recorta a [-30, 20], igual que en Stable Diffusion, para que
 * `exp` no se desborde al principio del entrenamiento. Donde el recorte actua
 * el gradiente respecto a `logvar` es cero, tanto por el sorteo como por la KL.
 *
 * El ruido se recibe hecho y no se genera dentro, por la misma razon que en los
 * muestreadores: poder repetir un calculo exacto y compararlo contra PyTorch.
 */
class GaussianaDiagonal {
 public:
  static constexpr float kLogVarMin = -30.0f;
  static constexpr float kLogVarMax = 20.0f;

  /**
   * @param params salida del codificador, `[N, 2C, h, w]`.
   * @param ruido `[N, C, h, w]`, normalmente N(0, 1).
   * @return el latente `z`, `[N, C, h, w]`.
   */
  Tensor Forward(const Tensor& params, const Tensor& ruido);

  /** @brief La media: el latente sin sortear, para codificar al evaluar. */
  [[nodiscard]] const Tensor& Media() const { return mu_; }
  /** @brief La log-varianza, ya recortada. */
  [[nodiscard]] const Tensor& LogVar() const { return logvar_; }

  /** @brief La KL del ultimo Forward, promediada en el lote. */
  [[nodiscard]] double KL() const;

  /**
   * @brief Gradiente respecto a `params` de `L_rec + peso_kl * KL`.
   *
   * @param dz gradiente de la perdida de reconstruccion respecto a `z`.
   * @param peso_kl el `beta` que pondera la KL. Se suman aqui dentro los dos
   *        caminos para que ningun entrenador pueda olvidarse de uno: con
   *        `peso_kl = 0` queda solo el de reconstruccion, y con `dz` a cero
   *        solo el de la KL.
   */
  [[nodiscard]] Tensor Backward(const Tensor& dz, float peso_kl) const;

 private:
  Tensor mu_, logvar_, sigma_, ruido_;
  std::vector<uint8_t> recortado_;   // 1 donde el recorte de logvar actuo
  int n_ = 0, c_ = 0;
  size_t espacial_ = 0;
};

}  // namespace latent
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LATENT_GAUSSIANA_H_
