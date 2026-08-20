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
#include "image/bitmap.h"
#include "image/bmp.h"
#include "image/jpeg.h"
#include "image/netpbm.h"
#include "image/png.h"
#include "image/renglones.h"
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
