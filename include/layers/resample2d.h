// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file resample2d.h
 * @brief Cambio de resolucion espacial: `Upsample2D` y `Downsample2D`.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_RESAMPLE2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_RESAMPLE2D_H_

#include <stdexcept>
#include <string>
#include <vector>

#include "../layer.h"

namespace neuralsuite {

/**
 * @class Upsample2D
 * @brief Duplica la resolucion repitiendo cada pixel (vecino mas proximo).
 *
 * `[N, C, H, W]` -> `[N, C, H*f, W*f]`. Es el bloque de subida de una U-Net.
 *
 * Hacia atras hay que **sumar**, no copiar: cada pixel de entrada aparece `f²`
 * veces en la salida, asi que recibe la suma de esos `f²` gradientes. Asignar
 * en vez de sumar deja un gradiente `f²` veces mas pequeno —o el de una esquina
 * arbitraria—, la red sigue entrenando algo peor, y no falla nada que no mire
 * el gradiente. Es el mismo error que en `Gather` con un token repetido.
 *
 * Va en pareja con `Downsample2D`, y las dos son adjuntas una de otra salvo el
 * factor `1/f²`: donde una copia, la otra suma; donde una promedia, la otra
 * reparte.
 */
class Upsample2D : public Layer {
 public:
  explicit Upsample2D(int factor = 2) : factor_(factor) {
    if (factor <= 0) {
      throw std::invalid_argument("Upsample2D: el factor debe ser positivo.");
    }
  }

  Tensor Forward(const Tensor& input) override;
  Tensor Backward(const Tensor& dout) override;

 private:
  int factor_;
  std::vector<int> forma_entrada_;
};

/**
 * @class Downsample2D
 * @brief Reduce la resolucion promediando bloques de `f x f`.
 *
 * `[N, C, H, W]` -> `[N, C, H/f, W/f]`. Es el bloque de bajada de una U-Net.
 *
 * Se promedia en vez de tomar el maximo —que es lo que hace `MaxPool2D`— porque
 * el maximo tira informacion de forma abrupta y su gradiente llega a un solo
 * pixel de cada bloque. En una red generativa eso deja huecos sin senal; el
 * promedio reparte el gradiente entre los `f²` de forma uniforme.
 *
 * Exige que `H` y `W` sean multiplos de `f`: un borde sobrante habria que
 * recortarlo o rellenarlo, y las dos opciones cambian el resultado en silencio.
 * Es mejor que aborte.
 */
class Downsample2D : public Layer {
 public:
  explicit Downsample2D(int factor = 2) : factor_(factor) {
    if (factor <= 0) {
      throw std::invalid_argument("Downsample2D: el factor debe ser positivo.");
    }
  }

  Tensor Forward(const Tensor& input) override;
  Tensor Backward(const Tensor& dout) override;

 private:
  int factor_;
  std::vector<int> forma_entrada_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_RESAMPLE2D_H_
