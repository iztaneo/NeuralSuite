// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file bitmap.h
 * @brief La imagen decodificada y las conversiones que necesita el modelo.
 *
 * Vive aparte porque lo necesitan a la vez los decodificadores y lo que va
 * despues de ellos. `Bitmap` estaba definido dentro de `png.h`, de modo que
 * `bmp.h` y `netpbm.h` incluian el decodificador de PNG entero solo para
 * usar esa estructura; y `ToGrayscale` y `Resize` estaban en la fachada, que
 * incluye a todos, asi que ningun decodificador podia usarlas. Al escribir la
 * separacion en renglones —que necesita las dos cosas— la dependencia se
 * volvio circular y hubo que ordenarlo.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_BITMAP_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_BITMAP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace neuralsuite {
namespace image {

/**
 * @struct Bitmap
 * @brief Imagen decodificada: 8 bits por canal, filas contiguas.
 *
 * `channels` es 1 (gris), 2 (gris + alfa), 3 (RGB) o 4 (RGBA).
 */
struct Bitmap {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  [[nodiscard]] bool Empty() const { return pixels.empty(); }
};

/**
 * @brief Pasa una imagen a un solo canal.
 *
 * Los coeficientes son los de la recomendacion BT.601, los mismos que usa
 * `Image.convert("L")` de Pillow: el ojo no percibe los tres canales por igual,
 * y promediarlos a partes iguales oscurece los verdes y aclara los azules. El
 * canal alfa se compone sobre blanco, que es lo que corresponde a un documento
 * escaneado: lo transparente es papel, no tinta.
 */
void ToGrayscale(const Bitmap& in, std::vector<float>* out);

/**
 * @brief Reescala una imagen en gris por interpolacion bilineal.
 *
 * Bilineal y no vecino mas proximo porque el CRNN reduce casi siempre —una
 * linea escaneada llega con varios cientos de pixeles de alto y el modelo
 * espera 32—, y con vecino mas proximo los trazos finos desaparecen segun donde
 * caiga la rejilla: la misma letra sale distinta segun su posicion.
 */
void Resize(const std::vector<float>& src, int src_w, int src_h,
                   std::vector<float>* dst, int dst_w, int dst_h);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_BITMAP_H_
