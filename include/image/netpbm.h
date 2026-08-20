// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file netpbm.h
 * @brief Decodificador de la familia Netpbm: PBM, PGM y PPM.
 *
 * Son los formatos mas simples que existen: una cabecera de texto con el numero
 * magico, las dimensiones y el valor maximo, y a continuacion los pixeles. Cada
 * uno tiene dos variantes, una en texto y otra binaria:
 *
 *     P1 / P4   blanco y negro, un bit por pixel
 *     P2 / P5   escala de grises
 *     P3 / P6   color RGB
 *
 * Estan aqui porque son el formato al que cualquier herramienta puede convertir
 * en una linea, y porque no dependen de nada: si algun dia falla el
 * decodificador de PNG, esta es la via para comprobar que el fallo es del PNG y
 * no de lo que hay despues.
 *
 * En PBM el 1 significa negro, al reves que en todos los demas. Es una
 * inversion facil de pasar por alto y da una imagen en negativo.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_NETPBM_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_NETPBM_H_

#include <cstdint>
#include <string>
#include <vector>
#include "bitmap.h"

namespace neuralsuite {
namespace image {
namespace detail {

/**
 * @brief Lee el siguiente entero de la cabecera, saltando espacios y comentarios.
 *
 * Un comentario empieza por `#` y llega hasta el fin de linea, y puede aparecer
 * en cualquier punto de la cabecera, incluso partiendo las dimensiones.
 */
inline bool NextHeaderInt(const uint8_t* data, size_t size, size_t* pos, int* value) {
  for (;;) {
    while (*pos < size && (data[*pos] == ' ' || data[*pos] == '\t' || data[*pos] == '\n' ||
                           data[*pos] == '\r')) {
      ++(*pos);
    }
    if (*pos < size && data[*pos] == '#') {
      while (*pos < size && data[*pos] != '\n') ++(*pos);
      continue;
    }
    break;
  }
  if (*pos >= size || data[*pos] < '0' || data[*pos] > '9') return false;
  int64_t n = 0;
  while (*pos < size && data[*pos] >= '0' && data[*pos] <= '9') {
    n = n * 10 + (data[(*pos)++] - '0');
    if (n > (1LL << 31)) return false;
  }
  *value = static_cast<int>(n);
  return true;
}

}  // namespace detail

/** @brief Decodifica un PBM, PGM o PPM, en su variante de texto o binaria. */
inline bool DecodeNetpbm(const uint8_t* data, size_t size, Bitmap* out, std::string* error) {
  using namespace detail;

  if (size < 2 || data[0] != 'P' || data[1] < '1' || data[1] > '6') {
    *error = "no es un archivo Netpbm: el numero magico no coincide";
    return false;
  }
  const int format = data[1] - '0';
  const bool binary = format >= 4;
  const int channels = (format == 3 || format == 6) ? 3 : 1;
  const bool bilevel = (format == 1 || format == 4);

  size_t pos = 2;
  int width = 0, height = 0, max_value = 1;
  if (!NextHeaderInt(data, size, &pos, &width) ||
      !NextHeaderInt(data, size, &pos, &height)) {
    *error = "no se pudieron leer las dimensiones";
    return false;
  }
  if (!bilevel && !NextHeaderInt(data, size, &pos, &max_value)) {
    *error = "no se pudo leer el valor maximo";
    return false;
  }
  if (width <= 0 || height <= 0 || max_value <= 0 || max_value > 65535) {
    *error = "la cabecera declara valores fuera de rango";
    return false;
  }
  if (static_cast<int64_t>(width) * height > 64LL * 1024 * 1024) {
    *error = "la imagen declara mas de 64 megapixeles";
    return false;
  }

  out->width = width;
  out->height = height;
  out->channels = channels;
  out->pixels.assign(static_cast<size_t>(width) * height * channels, 0);

  const bool wide = max_value > 255;  // dos bytes por muestra

  if (binary) {
    // Tras el ultimo campo de la cabecera va exactamente un separador.
    ++pos;
    if (bilevel) {
      const size_t stride = (static_cast<size_t>(width) + 7) / 8;
      if (size - pos < stride * static_cast<size_t>(height)) {
        *error = "faltan datos de pixeles";
        return false;
      }
      for (int y = 0; y < height; ++y) {
        const uint8_t* row = data + pos + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
          const int bit = (row[x / 8] >> (7 - x % 8)) & 1;
          // En PBM, 1 es negro.
          out->pixels[static_cast<size_t>(y) * width + x] = bit ? 0 : 255;
        }
      }
      return true;
    }
    const size_t samples = static_cast<size_t>(width) * height * channels;
    if (size - pos < samples * (wide ? 2u : 1u)) {
      *error = "faltan datos de pixeles";
      return false;
    }
    for (size_t i = 0; i < samples; ++i) {
      const uint32_t sample = wide ? static_cast<uint32_t>((data[pos + i * 2] << 8) |
                                                           data[pos + i * 2 + 1])
                                   : data[pos + i];
      out->pixels[i] = static_cast<uint8_t>(sample * 255u / static_cast<uint32_t>(max_value));
    }
    return true;
  }

  // Variante en texto. P1 es un caso aparte: cada pixel es un unico caracter
  // `0` o `1` y el separador es opcional, de modo que una fila se escribe
  // normalmente como `01101`. Leer eso con la misma rutina que los demas lo
  // interpretaria como el numero mil ciento uno. Se lee caracter a caracter.
  const size_t samples = static_cast<size_t>(width) * height * channels;
  if (bilevel) {
    for (size_t i = 0; i < samples; ++i) {
      while (pos < size && (data[pos] == '#' || data[pos] == ' ' || data[pos] == '\t' ||
                            data[pos] == '\n' || data[pos] == '\r')) {
        if (data[pos] == '#') {
          while (pos < size && data[pos] != '\n') ++pos;
        } else {
          ++pos;
        }
      }
      if (pos >= size || (data[pos] != '0' && data[pos] != '1')) {
        *error = "faltan valores de pixel: se leyeron " + std::to_string(i) + " de " +
                 std::to_string(samples);
        return false;
      }
      out->pixels[i] = (data[pos++] == '1') ? 0 : 255;  // en PBM, 1 es negro
    }
    return true;
  }

  for (size_t i = 0; i < samples; ++i) {
    int sample = 0;
    if (!NextHeaderInt(data, size, &pos, &sample)) {
      *error = "faltan valores de pixel: se leyeron " + std::to_string(i) + " de " +
               std::to_string(samples);
      return false;
    }
    if (sample > max_value) sample = max_value;
    out->pixels[i] = static_cast<uint8_t>(sample * 255 / max_value);
  }
  return true;
}

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_NETPBM_H_
