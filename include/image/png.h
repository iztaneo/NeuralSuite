// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file png.h
 * @brief Decodificador PNG completo (ISO/IEC 15948).
 *
 * Un PNG son trozos con nombre: `IHDR` describe la imagen, `PLTE` lleva la
 * paleta si la hay, uno o varios `IDAT` contienen los pixeles comprimidos con
 * zlib y `IEND` cierra. Cada trozo trae su CRC-32, que aqui se comprueba.
 *
 * Antes de comprimir, PNG filtra cada fila: la resta de la anterior, del pixel
 * de la izquierda, del promedio de ambos o del predictor Paeth. Eso no reduce
 * nada por si mismo, pero convierte los degradados en secuencias de valores
 * pequenos y repetidos, que es lo que DEFLATE sabe aprovechar. Deshacer el
 * filtro es la mitad de este archivo.
 *
 * Se admiten las profundidades 1, 2, 4, 8 y 16 bits, los cinco tipos de color y
 * el entrelazado Adam7. El entrelazado casi no se usa, pero decir "soporta PNG"
 * y fallar con uno de cada cien archivos es peor que no decirlo. Las imagenes de
 * 16 bits se reducen a 8: el modelo trabaja en float32 normalizado y no
 * distingue esa diferencia.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_PNG_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_PNG_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "bitmap.h"
#include "inflate.h"

namespace neuralsuite {
namespace image {

namespace detail {

/** @brief CRC-32 con el polinomio de PNG, calculado sin tabla precompilada. */
uint32_t Crc32(const uint8_t* data, size_t size, uint32_t crc = 0xFFFFFFFFu);

uint32_t ReadBigEndian32(const uint8_t* p);

/**
 * @brief Predictor Paeth: elige el vecino que menos se aparta de a + b - c.
 *
 * Es el filtro que mejor funciona en fotografias, y el unico de los cinco que
 * no es una resta directa.
 */
uint8_t PaethPredictor(int a, int b, int c);

/** @brief Deshace el filtro de una fila, que depende de la ya reconstruida. */
bool UnfilterRow(uint8_t filter, uint8_t* row, const uint8_t* prev, size_t bytes,
                        size_t bpp, std::string* error);

/** @brief Los siete pasos de Adam7: origen y salto de cada uno. */
struct Adam7Pass {
  int x_start, y_start, x_step, y_step;
};
inline const Adam7Pass kAdam7[7] = {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4},
                                    {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}};

/** @brief Extrae la muestra `index` de una fila empaquetada a `depth` bits. */
uint16_t ReadSample(const uint8_t* row, size_t index, int depth);

}  // namespace detail

/**
 * @brief Decodifica un PNG en memoria.
 *
 * Devuelve false y un mensaje en `error` ante cualquier archivo que no cumpla la
 * norma. La entrada se trata como hostil: ningun campo declarado por el archivo
 * se usa para dimensionar memoria sin comprobarlo antes.
 */
bool DecodePng(const uint8_t* data, size_t size, Bitmap* out, std::string* error);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_PNG_H_
