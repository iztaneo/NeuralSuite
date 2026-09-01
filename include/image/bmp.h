// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file bmp.h
 * @brief Decodificador BMP (mapa de bits de Windows).
 *
 * BMP no comprime, asi que aqui no hay nada parecido a `inflate`. La dificultad
 * esta en otro sitio: el formato lleva treinta anos acumulando variantes y casi
 * todas siguen apareciendo. Se admiten las dos cabeceras habituales —la de OS/2
 * de 12 bytes y la de Windows de 40, que las versiones 4 y 5 solo extienden—,
 * las profundidades de 1, 4, 8, 16, 24 y 32 bits, con paleta o sin ella, y las
 * filas en cualquiera de los dos sentidos.
 *
 * Tres detalles que son fuente habitual de errores:
 *
 * - Las filas se rellenan hasta un multiplo de cuatro bytes. Sin eso, una imagen
 *   cuyo ancho no sea multiplo de cuatro sale inclinada.
 * - El origen esta abajo. Un alto negativo, y solo entonces, significa que las
 *   filas vienen de arriba abajo.
 * - Los canales van en orden BGR, no RGB.
 *
 * La salida es siempre RGB, o RGBA si el archivo declara canal alfa. Incluso un
 * BMP de un bit sale con tres canales: sus dos colores vienen de una paleta y
 * pueden ser cualesquiera, de modo que tratarlo como escala de grises seria
 * suponer algo que el formato no garantiza. Un PNG bilevel, en cambio, si es
 * gris por definicion y sale con un solo canal.
 *
 * Las variantes comprimidas RLE4 y RLE8 se rechazan con un mensaje explicito.
 * No es que sean dificiles: es que no hay con que comprobarlas —Pillow no las
 * escribe— y prefiero no incluir codigo que nadie ha verificado.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_BMP_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_BMP_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "bitmap.h"

namespace neuralsuite {
namespace image {
namespace detail {

uint16_t ReadLittleEndian16(const uint8_t* p);
uint32_t ReadLittleEndian32(const uint8_t* p);

/** @brief Desplazamiento y anchura de una mascara de bits contigua. */
void MaskShiftWidth(uint32_t mask, int* shift, int* width);

}  // namespace detail

/** @brief Decodifica un BMP en memoria. */
bool DecodeBmp(const uint8_t* data, size_t size, Bitmap* out, std::string* error);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_BMP_H_
