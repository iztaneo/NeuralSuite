// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file resblock.h
 * @brief Bloque residual condicionado por el paso de difusion.
 */

#ifndef NEURAL_SUITE_INCLUDE_DIFFUSION_RESBLOCK_H_
#define NEURAL_SUITE_INCLUDE_DIFFUSION_RESBLOCK_H_

#include <memory>
#include <stdexcept>
#include <string>

#include "../layer.h"
#include "../layers/conv2d.h"
#include "../layers/groupnorm.h"
#include "../layers/linear.h"

namespace neuralsuite {
namespace diffusion {

/**
 * @class ResBlockTiempo
 * @brief La pieza que se repite por toda la U-Net.
 *
 *     x ──► GroupNorm ──► SiLU ──► Conv2D ──┐
 *                                            │
 *     t ──► SiLU ──► Linear ──────► (+) ─────┤   se suma por canal
 *                                            │
 *                       GroupNorm ──► SiLU ──► Conv2D ──┐
 *                                                        │
 *     x ─────────────► atajo ──────────────────────────► (+) ──► salida
 *
 * Tres decisiones que no son evidentes:
 *
 * 1. **El tiempo se suma por canal, no por pixel.** La proyeccion lineal
 *    convierte el embedding en un vector de `canales_salida` valores, y cada uno
 *    se suma a todo su mapa. Es lo correcto: el paso de difusion es una
 *    propiedad de la imagen entera, no de cada posicion. Sumarlo por pixel
 *    exigiria una proyeccion del tamano del mapa y el bloque dejaria de servir
 *    para resoluciones distintas.
 *
 * 2. **El atajo lleva convolucion solo si cambian los canales.** Cuando entran y
 *    salen los mismos, el atajo es la identidad, que es lo que hace que el
 *    gradiente llegue intacto a las capas de abajo. Meter una convolucion 1x1
 *    innecesaria funcionaria igual pero desperdicia el atajo.
 *
 * 3. **La normalizacion va ANTES de la activacion y de la convolucion**
 *    —«pre-norm»—, igual que en el transformer del proyecto. Con la norma
 *    despues, las redes profundas cuestan mas de entrenar, y esto va a apilarse
 *    muchas veces.
 */
class ResBlockTiempo : public Layer {
 public:
  ResBlockTiempo(int canales_entrada, int canales_salida, int dim_tiempo,
                 int grupos = 8);

  /** @brief `[N, Cin, H, W]` con `[N, dim_tiempo]` -> `[N, Cout, H, W]`. */
  Tensor Forward(const Tensor& x, const Tensor& t_emb);

  /** @brief Sin tiempo no hay bloque: la interfaz de `Layer` no basta aqui. */
  Tensor Forward(const Tensor& x) override {
    throw std::logic_error(
        "ResBlockTiempo: hace falta el embedding del paso; usa Forward(x, t_emb).");
  }

  /** @brief Devuelve el gradiente de `x`; el del tiempo va en `GradTiempo()`. */
  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Tensor& GradTiempo() const { return d_t_emb_; }

  GroupNormLayer& Norm1() { return norm1_; }
  GroupNormLayer& Norm2() { return norm2_; }
  Conv2D& Conv1() { return conv1_; }
  Conv2D& Conv2() { return conv2_; }
  Linear& ProyTiempo() { return proy_t_; }
  Conv2D* Atajo() { return atajo_.get(); }

 private:
  // Solo lo que el forward y el backward vuelven a necesitar. `c_in` no se
  // guarda: su unico uso —decidir si hace falta el atajo— se resuelve en el
  // constructor y no vuelve a hacer falta.
  int c_out_, dim_t_;

  GroupNormLayer norm1_, norm2_;
  Conv2D conv1_, conv2_;
  Linear proy_t_;
  std::unique_ptr<Conv2D> atajo_;   // solo si c_in != c_out

  // Lo que el backward necesita del forward.
  Tensor x_, h1_pre_, h1_act_, t_pre_, t_act_, h2_, h2_pre_, h2_act_;
  Tensor d_t_emb_;
  int n_ = 0, h_ = 0, w_ = 0;
};

}  // namespace diffusion
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DIFFUSION_RESBLOCK_H_
