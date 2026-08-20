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
inline void ToGrayscale(const Bitmap& in, std::vector<float>* out) {
  out->assign(static_cast<size_t>(in.width) * in.height, 0.0f);
  const int c = in.channels;
  for (size_t i = 0; i < out->size(); ++i) {
    const uint8_t* p = in.pixels.data() + i * c;
    float value;
    if (c >= 3) {
      value = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    } else {
      value = p[0];
    }
    if (c == 2 || c == 4) {
      const float alpha = p[c - 1] / 255.0f;
      value = value * alpha + 255.0f * (1.0f - alpha);
    }
    (*out)[i] = value / 255.0f;
  }
}

/**
 * @brief Reescala una imagen en gris por interpolacion bilineal.
 *
 * Bilineal y no vecino mas proximo porque el CRNN reduce casi siempre —una
 * linea escaneada llega con varios cientos de pixeles de alto y el modelo
 * espera 32—, y con vecino mas proximo los trazos finos desaparecen segun donde
 * caiga la rejilla: la misma letra sale distinta segun su posicion.
 */
inline void Resize(const std::vector<float>& src, int src_w, int src_h,
                   std::vector<float>* dst, int dst_w, int dst_h) {
  dst->assign(static_cast<size_t>(dst_w) * dst_h, 0.0f);
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  // Se mapean centros de pixel, no esquinas: con esquinas la imagen se desplaza
  // medio pixel y el borde derecho se estira.
  const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
  const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    const float fy = std::max(0.0f, (y + 0.5f) * scale_y - 0.5f);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = fy - static_cast<float>(y0);

    for (int x = 0; x < dst_w; ++x) {
      const float fx = std::max(0.0f, (x + 0.5f) * scale_x - 0.5f);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = fx - static_cast<float>(x0);

      const float a = src[static_cast<size_t>(y0) * src_w + x0];
      const float b = src[static_cast<size_t>(y0) * src_w + x1];
      const float c = src[static_cast<size_t>(y1) * src_w + x0];
      const float d = src[static_cast<size_t>(y1) * src_w + x1];

      (*dst)[static_cast<size_t>(y) * dst_w + x] =
          a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy;
    }
  }
}

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_BITMAP_H_
