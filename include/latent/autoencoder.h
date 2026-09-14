// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file autoencoder.h
 * @brief Codificador y decodificador convolucionales del autoencoder del LDM-1.
 */

#ifndef NEURAL_SUITE_INCLUDE_LATENT_AUTOENCODER_H_
#define NEURAL_SUITE_INCLUDE_LATENT_AUTOENCODER_H_

#include "../layer.h"
#include "../layers/conv2d.h"
#include "../layers/groupnorm.h"
#include "../layers/resample2d.h"
#include "../layers/resblock2d.h"

namespace neuralsuite {
namespace latent {

/**
 * @class Codificador
 * @brief Imagen `[N, 1, H, W]` -> `[N, 2*C, H/4, W/4]`.
 *
 *     conv 3×3 → ResBlock(c) → baja → ResBlock(c→2c) → baja → ResBlock(2c)
 *              → GroupNorm → SiLU → conv 3×3 → 2·C canales
 *
 * Reduce por `f = 4` con dos `Downsample2D`. Con MNIST rellenado a 32×32 el
 * latente queda en 8×8, que es multiplo de 4 y por tanto lo acepta `UNet2D`;
 * con 28×28 habria quedado en 7×7 y no.
 *
 * La salida tiene el DOBLE de los canales del latente: la mitad es la media y
 * la otra mitad el logaritmo de la varianza de la gaussiana de la que se
 * muestrea el latente. El reparto y el muestreo son el escalon siguiente; aqui
 * solo se garantiza que el codificador produce esos numeros bien cableado.
 */
class Codificador : public Layer {
 public:
  /**
   * @param canales_latente `C`, canales del latente.
   * @param canales anchura base; el nivel bajo usa `2 * canales`.
   */
  Codificador(int canales_imagen, int canales, int canales_latente, int grupos);

  Tensor Forward(const Tensor& x) override;
  Tensor Backward(const Tensor& dout) override;

  /** @name Submodulos, para cargar los pesos de la referencia por nombre. */
  ///@{
  Conv2D& ConvEntrada() { return conv_entrada_; }
  ResBlock2D& Res0() { return res0_; }
  ResBlock2D& Res1() { return res1_; }
  ResBlock2D& Res2() { return res2_; }
  GroupNormLayer& NormSalida() { return norm_salida_; }
  Conv2D& ConvSalida() { return conv_salida_; }
  ///@}

 private:
  int canales_imagen_;
  Conv2D conv_entrada_;
  ResBlock2D res0_, res1_, res2_;
  Downsample2D bajar0_, bajar1_;
  GroupNormLayer norm_salida_;
  Conv2D conv_salida_;
  Tensor pre_salida_, act_salida_;
};

/**
 * @class Decodificador
 * @brief Latente `[N, C, h, w]` -> imagen `[N, 1, 4h, 4w]`.
 *
 *     conv 3×3 → ResBlock(2c) → sube → ResBlock(2c→c) → sube → ResBlock(c)
 *              → GroupNorm → SiLU → conv 3×3 → 1 canal
 *
 * El espejo del codificador. Sube con `Upsample2D` (vecino mas proximo) seguido
 * de convoluciones en vez de con una convolucion transpuesta: es como lo hace el
 * decodificador del paper, y evita el patron de tablero de ajedrez que la
 * transpuesta produce cuando el nucleo no es multiplo del paso.
 */
class Decodificador : public Layer {
 public:
  Decodificador(int canales_latente, int canales, int canales_imagen, int grupos);

  Tensor Forward(const Tensor& z) override;
  Tensor Backward(const Tensor& dout) override;

  /** @name Submodulos, para cargar los pesos de la referencia por nombre. */
  ///@{
  Conv2D& ConvEntrada() { return conv_entrada_; }
  ResBlock2D& Res0() { return res0_; }
  ResBlock2D& Res1() { return res1_; }
  ResBlock2D& Res2() { return res2_; }
  GroupNormLayer& NormSalida() { return norm_salida_; }
  Conv2D& ConvSalida() { return conv_salida_; }
  ///@}

 private:
  int canales_latente_;
  Conv2D conv_entrada_;
  ResBlock2D res0_, res1_, res2_;
  Upsample2D subir0_, subir1_;
  GroupNormLayer norm_salida_;
  Conv2D conv_salida_;
  Tensor pre_salida_, act_salida_;
};

}  // namespace latent
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LATENT_AUTOENCODER_H_
