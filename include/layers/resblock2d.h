// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file resblock2d.h
 * @brief Bloque residual convolucional sin condicionar.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_RESBLOCK2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_RESBLOCK2D_H_

#include <memory>

#include "../layer.h"
#include "conv2d.h"
#include "groupnorm.h"

namespace neuralsuite {

/**
 * @class ResBlock2D
 * @brief El bloque del que se construyen el codificador y el decodificador.
 *
 *     x ──► GroupNorm ──► SiLU ──► Conv 3×3 ──► GroupNorm ──► SiLU ──► Conv 3×3 ──(+)──► y
 *     └────────────────────── atajo (identidad, o Conv 1×1) ─────────────────────┘
 *
 * Es `ResBlockTiempo` sin la inyeccion del paso de difusion: un autoencoder no
 * sabe de pasos. Se escribe aparte en vez de hacer opcional el tiempo en aquel
 * porque aquel ya esta verificado y su backward tiene cuatro ramas; tocarlo
 * para quitarle una arriesgaba lo que ya funciona a cambio de ahorrar cien
 * lineas.
 *
 * El atajo solo lleva convolucion cuando cambian los canales. Si no, es la
 * identidad, y por ahi el gradiente llega intacto a las capas de abajo, que es
 * para lo que existe el residuo.
 */
class ResBlock2D : public Layer {
 public:
  ResBlock2D(int canales_entrada, int canales_salida, int grupos);

  /** @brief `[N, Cin, H, W]` -> `[N, Cout, H, W]`. */
  Tensor Forward(const Tensor& x) override;

  /** @brief Devuelve el gradiente respecto a `x` y acumula el de los pesos. */
  Tensor Backward(const Tensor& dout) override;

  /** @name Submodulos, para cargar los pesos de la referencia por nombre. */
  ///@{
  GroupNormLayer& Norm1() { return norm1_; }
  GroupNormLayer& Norm2() { return norm2_; }
  Conv2D& Conv1() { return conv1_; }
  Conv2D& Conv2() { return conv2_; }
  Conv2D* Atajo() { return atajo_.get(); }
  ///@}

 private:
  GroupNormLayer norm1_, norm2_;
  Conv2D conv1_, conv2_;
  std::unique_ptr<Conv2D> atajo_;   // solo si cambian los canales

  // Lo que el backward necesita del forward.
  Tensor h1_pre_, h1_act_, h2_pre_, h2_act_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_RESBLOCK2D_H_
