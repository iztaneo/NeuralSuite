// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file image.h
 * @brief Carga de imagenes desde disco y conversion a tensor.
 *
 * Reune los decodificadores de PNG, JPEG, BMP y Netpbm bajo una sola llamada y
 * anade lo que el modelo necesita despues: pasar a gris, normalizar y ajustar el
 * alto.
 *
 * El formato se decide por los primeros bytes del archivo, nunca por su
 * extension. Un `.png` que en realidad es un BMP es algo corriente —lo produce
 * cualquier renombrado— y adivinarlo por el nombre convierte un archivo
 * perfectamente legible en un error incomprensible.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_H_

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "image/bmp.h"
#include "image/jpeg.h"
#include "image/netpbm.h"
#include "image/png.h"
#include "tensor.h"

namespace neuralsuite {
namespace image {

/** @brief Formatos que se reconocen. */
enum class Format { kUnknown, kPng, kBmp, kJpeg, kNetpbm };

/** @brief Identifica el formato por el numero magico del archivo. */
inline Format DetectFormat(const uint8_t* data, size_t size) {
  if (size >= 8 && data[0] == 0x89 && std::memcmp(data + 1, "PNG", 3) == 0) return Format::kPng;
  if (size >= 2 && data[0] == 'B' && data[1] == 'M') return Format::kBmp;
  if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) return Format::kJpeg;
  if (size >= 2 && data[0] == 'P' && data[1] >= '1' && data[1] <= '6') return Format::kNetpbm;
  return Format::kUnknown;
}

inline const char* FormatName(Format format) {
  switch (format) {
    case Format::kPng: return "PNG";
    case Format::kBmp: return "BMP";
    case Format::kJpeg: return "JPEG";
    case Format::kNetpbm: return "Netpbm";
    default: return "desconocido";
  }
}

/** @brief Decodifica una imagen ya cargada en memoria. */
inline bool Decode(const uint8_t* data, size_t size, Bitmap* out, std::string* error) {
  switch (DetectFormat(data, size)) {
    case Format::kPng: return DecodePng(data, size, out, error);
    case Format::kBmp: return DecodeBmp(data, size, out, error);
    case Format::kJpeg: return DecodeJpeg(data, size, out, error);
    case Format::kNetpbm: return DecodeNetpbm(data, size, out, error);
    default:
      *error = "formato de imagen no reconocido. Se admiten PNG, JPEG, BMP y Netpbm (PBM/PGM/PPM).";
      return false;
  }
}

/** @brief Lee un archivo del disco y lo decodifica. */
inline bool Load(const std::string& path, Bitmap* out, std::string* error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "no se pudo abrir '" + path + "'";
    return false;
  }
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
  if (data.empty()) {
    *error = "'" + path + "' esta vacio";
    return false;
  }
  return Decode(data.data(), data.size(), out, error);
}

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

/**
 * @brief Carga una imagen como tensor `[1, 1, target_height, ancho]` en gris.
 *
 * El ancho se deduce conservando la proporcion y se redondea al multiplo de
 * `width_multiple` que exige la red. `invert` cambia el criterio de que es
 * tinta: el CRNN se entrena con trazo claro sobre fondo oscuro, mientras que un
 * documento escaneado llega al reves.
 */
inline bool LoadAsTensor(const std::string& path, int target_height, int width_multiple,
                         bool invert, Tensor* out, std::string* error) {
  Bitmap bitmap;
  if (!Load(path, &bitmap, error)) return false;

  std::vector<float> gray;
  ToGrayscale(bitmap, &gray);

  int width = static_cast<int>(std::lround(static_cast<double>(bitmap.width) * target_height /
                                           std::max(1, bitmap.height)));
  width = std::max(width_multiple, width);
  width = ((width + width_multiple - 1) / width_multiple) * width_multiple;

  std::vector<float> scaled;
  Resize(gray, bitmap.width, bitmap.height, &scaled, width, target_height);

  out->Resize({1, 1, target_height, width});
  for (size_t i = 0; i < scaled.size(); ++i) {
    (*out)[i] = invert ? 1.0f - scaled[i] : scaled[i];
  }
  return true;
}

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_H_
