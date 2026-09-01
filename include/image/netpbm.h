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
bool NextHeaderInt(const uint8_t* data, size_t size, size_t* pos, int* value);

}  // namespace detail

/** @brief Decodifica un PBM, PGM o PPM, en su variante de texto o binaria. */
bool DecodeNetpbm(const uint8_t* data, size_t size, Bitmap* out, std::string* error);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_NETPBM_H_
