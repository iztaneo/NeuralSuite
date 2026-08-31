// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file jpeg.h
 * @brief Decodificador JPEG: secuencial de linea base y progresivo (ITU-T T.81).
 *
 * JPEG es, con diferencia, el mas complicado de los formatos que lee este
 * proyecto, y por una razon distinta de la que parece. La dificultad no esta en
 * la transformada, que son cuarenta lineas, sino en que el archivo describe un
 * proceso y no unos pixeles: cuantas tablas hay, en que orden se recorren los
 * bloques, cuantos scans hacen falta para completar un coeficiente.
 *
 * La cadena, de dentro afuera:
 *
 *  1. Los coeficientes van codificados con Huffman, y el flujo lleva relleno:
 *     un 0xFF dentro de los datos se escribe como 0xFF 0x00, porque 0xFF
 *     empieza un marcador.
 *  2. Cada bloque de 8x8 se recorre en zigzag, de las frecuencias bajas a las
 *     altas, y se codifica por rachas de ceros.
 *  3. Los coeficientes se multiplican por la tabla de cuantizacion. Ahi es
 *     donde JPEG pierde informacion, y es irreversible.
 *  4. La transformada inversa del coseno devuelve los 64 pixeles.
 *  5. El color va en YCbCr, y la crominancia suele venir a la mitad de
 *     resolucion. Hay que interpolarla.
 *
 * En progresivo la imagen se reparte entre varios scans: uno trae los bits
 * altos del coeficiente continuo, otro un rango de frecuencias, otro refina un
 * bit mas de los anteriores. Por eso este decodificador acumula todos los
 * coeficientes primero y solo transforma al final: es lo que permite tratar los
 * dos modos con el mismo codigo en vez de escribir dos decodificadores.
 *
 * Una advertencia que conviene tener presente al comparar con otra
 * implementacion: la salida de un JPEG no esta especificada bit a bit. La norma
 * fija requisitos de precision para la transformada inversa (T.83), no un
 * resultado exacto, de modo que dos decodificadores correctos difieren en
 * algunos pixeles por una unidad. Eso no es un defecto de ninguno de los dos.
 * Aqui la transformada se comprueba contra su definicion matematica en doble
 * precision, que es la referencia que si es exacta.
 *
 * No se admiten: el modo aritmetico (patentado en su dia, practicamente
 * inexistente), el sin perdida, el jerarquico, ni los JPEG de 12 bits. Todos se
 * rechazan con un mensaje que dice cual es.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_JPEG_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_JPEG_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "bitmap.h"

namespace neuralsuite {
namespace image {
namespace jpeg_detail {

/** @brief Orden zigzag: del coeficiente continuo a las frecuencias altas. */
inline const uint8_t kZigZag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

/**
 * @class HuffmanTable
 * @brief Codigo canonico de JPEG, decodificado por longitud creciente.
 *
 * Igual que en DEFLATE, el codigo queda determinado por cuantos simbolos hay de
 * cada longitud. La diferencia es el orden de los bits: JPEG los lee del mas
 * significativo al menos, al reves que zlib.
 */
struct HuffmanTable {
  std::array<int32_t, 17> min_code{};
  std::array<int32_t, 17> max_code{};
  std::array<int32_t, 17> val_index{};
  std::vector<uint8_t> values;
  bool defined = false;

  bool Build(const uint8_t counts[16], const uint8_t* symbols, size_t symbol_count) {
    values.assign(symbols, symbols + symbol_count);
    int32_t code = 0;
    size_t index = 0;
    for (int len = 1; len <= 16; ++len) {
      val_index[len] = static_cast<int32_t>(index) - code;
      min_code[len] = code;
      code += counts[len - 1];
      index += counts[len - 1];
      max_code[len] = code - 1;
      if (counts[len - 1] == 0) max_code[len] = -1;  // no hay codigos de esta longitud
      code <<= 1;
    }
    if (index != symbol_count) return false;
    defined = true;
    return true;
  }
};

/**
 * @class EntropyReader
 * @brief Lee bits del segmento comprimido, deshaciendo el relleno de 0xFF.
 *
 * Cuando se agotan los datos devuelve ceros en vez de fallar. Es lo que hace
 * libjpeg y no es indulgencia gratuita: un archivo truncado por el final es el
 * caso mas comun de JPEG danado, y rendir la parte que si esta bien vale mas
 * que rechazarlo entero.
 */
class EntropyReader {
 public:
  EntropyReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  int ReadBit() {
    if (bits_ == 0) {
      if (pos_ >= size_) { exhausted_ = true; return 0; }
      byte_ = data_[pos_++];
      if (byte_ == 0xFF) {
        // 0xFF 0x00 es un 0xFF literal; 0xFF seguido de otra cosa es un marcador
        // y ahi se acaba el segmento.
        if (pos_ < size_ && data_[pos_] == 0x00) {
          ++pos_;
        } else {
          exhausted_ = true;
          return 0;
        }
      }
      bits_ = 8;
    }
    --bits_;
    return (byte_ >> bits_) & 1;
  }

  int Receive(int count) {
    int value = 0;
    for (int i = 0; i < count; ++i) value = (value << 1) | ReadBit();
    return value;
  }

  /** @brief Convierte la magnitud codificada en su valor con signo. */
  static int Extend(int value, int count) {
    if (count == 0) return 0;
    return value < (1 << (count - 1)) ? value - (1 << count) + 1 : value;
  }

  int Decode(const HuffmanTable& table) {
    int32_t code = 0;
    for (int len = 1; len <= 16; ++len) {
      code = (code << 1) | ReadBit();
      if (table.max_code[len] >= 0 && code <= table.max_code[len]) {
        const size_t index = static_cast<size_t>(table.val_index[len] + code);
        if (index >= table.values.size()) return -1;
        return table.values[index];
      }
    }
    return -1;
  }

  /** @brief Descarta los bits sueltos y salta el marcador de reinicio. */
  void RestartAt(size_t position) {
    pos_ = position;
    bits_ = 0;
    exhausted_ = false;
  }

  [[nodiscard]] size_t Position() const { return pos_; }
  [[nodiscard]] bool Exhausted() const { return exhausted_; }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
  uint8_t byte_ = 0;
  int bits_ = 0;
  bool exhausted_ = false;
};

/**
 * @brief Transformada inversa del coseno de 8x8, separable.
 *
 * Se aplica una transformada de una dimension a las filas y otra a las
 * columnas, que da el mismo resultado que la de dos dimensiones con muchas
 * menos operaciones. Se calcula en `double` y con la tabla de cosenos exacta:
 * el objetivo aqui no es ser rapido sino no introducir error propio, para que
 * cualquier diferencia con otro decodificador sea suya y no nuestra.
 */
void InverseDct(const int32_t* input, uint8_t* output, int stride);

/** @brief Un componente de color con su rejilla de coeficientes. */
struct Component {
  int id = 0;
  int h = 1, v = 1;          // factores de muestreo
  int quant_table = 0;
  int dc_table = 0, ac_table = 0;
  int blocks_per_line = 0;   // rejilla real, sin el relleno del MCU
  int blocks_per_column = 0;
  int stride_blocks = 0;     // rejilla reservada, multiplo del MCU
  int rows_blocks = 0;
  std::vector<int32_t> coefficients;
  std::vector<uint8_t> samples;  // pixeles ya transformados, antes de interpolar
  int dc_pred = 0;
};

}  // namespace jpeg_detail

/**
 * @brief Decodifica un JPEG en memoria.
 *
 * Devuelve 1 canal para las imagenes en gris y 3 para las de color.
 */
bool DecodeJpeg(const uint8_t* data, size_t size, Bitmap* out, std::string* error);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_JPEG_H_
