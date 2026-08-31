// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file inflate.h
 * @brief Descompresor DEFLATE (RFC 1951) y su envoltorio zlib (RFC 1950).
 *
 * Existe porque PNG guarda sus pixeles comprimidos con zlib y el proyecto no
 * enlaza bibliotecas externas. Es la pieza que hacia falta para que `ocr_cli`
 * lea de verdad el archivo que se le pasa en vez de anunciar que no puede.
 *
 * DEFLATE alterna dos ideas. La primera es LZ77: en vez de repetir una secuencia
 * ya vista, se emite "retrocede D bytes y copia L". La segunda es Huffman: los
 * simbolos frecuentes se codifican con menos bits. Un flujo son bloques, y cada
 * bloque declara si va sin comprimir, con un arbol fijo definido en la norma, o
 * con un arbol propio que viene descrito al principio del bloque.
 *
 * La decodificacion Huffman es la canonica de `puff.c`: en lugar de construir
 * una tabla de busqueda, se recorren las longitudes de codigo contando cuantos
 * codigos hay de cada una. Es mas lento por simbolo y mucho mas corto de
 * escribir y de leer, y aqui descomprimir una imagen no esta en ningun bucle
 * critico.
 *
 * Todo el codigo trata la entrada como hostil: un archivo mal formado —o
 * fabricado a proposito— tiene que devolver un error, nunca leer fuera de su
 * buffer ni reservar memoria sin limite.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_INFLATE_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_INFLATE_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace neuralsuite {
namespace image {
namespace detail {

/** @brief Lector de bits en el orden que usa DEFLATE: del menos significativo. */
class BitReader {
 public:
  BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  /** @brief Lee `count` bits. Devuelve false si el flujo se acaba antes. */
  bool Bits(int count, uint32_t* out) {
    uint32_t value = buffer_;
    while (bits_ < count) {
      if (pos_ >= size_) return false;
      value |= static_cast<uint32_t>(data_[pos_++]) << bits_;
      bits_ += 8;
    }
    buffer_ = value >> count;
    bits_ -= count;
    *out = value & ((1u << count) - 1u);
    return true;
  }

  /** @brief Descarta los bits sueltos y vuelve a un limite de byte. */
  void AlignToByte() {
    buffer_ = 0;
    bits_ = 0;
  }

  bool ReadBytes(uint8_t* dst, size_t n) {
    if (size_ - pos_ < n) return false;
    std::memcpy(dst, data_ + pos_, n);
    pos_ += n;
    return true;
  }

  [[nodiscard]] size_t Position() const { return pos_; }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
  uint32_t buffer_ = 0;
  int bits_ = 0;
};

/** @brief Longitud maxima de un codigo Huffman en DEFLATE. */
constexpr int kMaxCodeBits = 15;

/**
 * @class HuffmanTable
 * @brief Codigo canonico descrito solo por la longitud de cada simbolo.
 *
 * Un codigo canonico queda determinado por cuantos bits ocupa cada simbolo: no
 * hace falta transmitir los codigos, solo las longitudes. Se guarda cuantos
 * codigos hay de cada longitud y los simbolos ordenados, que es lo justo para
 * decodificar recorriendo las longitudes de menor a mayor.
 */
class HuffmanTable {
 public:
  bool Build(const uint8_t* lengths, int count) {
    counts_.fill(0);
    for (int i = 0; i < count; ++i) {
      if (lengths[i] > kMaxCodeBits) return false;
      ++counts_[lengths[i]];
    }
    // Las longitudes 0 marcan simbolos que no aparecen.
    counts_[0] = 0;

    // Un codigo es incompleto si sobran combinaciones y sobrepasado si faltan.
    // El unico incompleto que la norma permite es el de un solo simbolo.
    int left = 1;
    for (int len = 1; len <= kMaxCodeBits; ++len) {
      left <<= 1;
      left -= counts_[len];
      if (left < 0) return false;
    }

    std::array<uint16_t, kMaxCodeBits + 1> offsets{};
    for (int len = 1; len < kMaxCodeBits; ++len) {
      offsets[len + 1] = static_cast<uint16_t>(offsets[len] + counts_[len]);
    }
    symbols_.assign(static_cast<size_t>(count), 0);
    for (int i = 0; i < count; ++i) {
      if (lengths[i] != 0) symbols_[offsets[lengths[i]]++] = static_cast<uint16_t>(i);
    }
    return true;
  }

  /** @brief Decodifica un simbolo. Devuelve -1 si el codigo no es valido. */
  [[nodiscard]] int Decode(BitReader& reader) const {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= kMaxCodeBits; ++len) {
      uint32_t bit = 0;
      if (!reader.Bits(1, &bit)) return -1;
      code |= static_cast<int>(bit);
      const int count = counts_[len];
      if (code - first < count) return symbols_[static_cast<size_t>(index + (code - first))];
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    return -1;
  }

 private:
  std::array<uint16_t, kMaxCodeBits + 1> counts_{};
  std::vector<uint16_t> symbols_;
};

// Tablas de la norma: para cada simbolo de longitud (257..285) y de distancia
// (0..29), cuanto vale la base y cuantos bits extra la acompanan.
inline const uint16_t kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,
                                         15, 17, 19, 23, 27, 31, 35, 43, 51,  59,
                                         67, 83, 99, 115, 131, 163, 195, 227, 258};
inline const uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                         2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline const uint16_t kDistBase[30] = {1,    2,    3,    4,    5,    7,     9,     13,
                                       17,   25,   33,   49,   65,   97,    129,   193,
                                       257,  385,  513,  769,  1025, 1537,  2049,  3073,
                                       4097, 6145, 8193, 12289, 16385, 24577};
inline const uint8_t kDistExtra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                        4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                        9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/** @brief Arbol fijo del tipo de bloque 01, definido en la propia norma. */
void BuildFixedTables(HuffmanTable* literals, HuffmanTable* distances);

/**
 * @brief Lee el arbol propio de un bloque de tipo 10.
 *
 * Las longitudes de codigo se transmiten a su vez comprimidas con un tercer
 * arbol, cuyas longitudes vienen en un orden fijo pensado para que las mas
 * probables queden delante y las ultimas puedan omitirse.
 */
bool ReadDynamicTables(BitReader& reader, HuffmanTable* literals,
                              HuffmanTable* distances, std::string* error);

/**
 * @brief Descomprime un flujo DEFLATE crudo, sin la cabecera de zlib.
 *
 * `max_output` acota cuanto se puede llegar a producir: sin ese limite, un
 * archivo pequeno y malicioso puede pedir gigabytes de memoria.
 */
bool Inflate(const uint8_t* data, size_t size, size_t max_output,
                    std::vector<uint8_t>* out, std::string* error);

/** @brief Suma de comprobacion Adler-32, la que lleva la envoltura zlib. */
uint32_t Adler32(const uint8_t* data, size_t size);

/**
 * @brief Descomprime un flujo zlib: cabecera de dos bytes, DEFLATE y Adler-32.
 *
 * La suma se comprueba de verdad. Aceptarla sin mirar dejaria pasar en silencio
 * un archivo corrupto, que es como se cuelan los defectos dificiles de ubicar:
 * el error aparece mucho despues, en el modelo, y parece cosa del modelo.
 */
bool ZlibInflate(const uint8_t* data, size_t size, size_t max_output,
                        std::vector<uint8_t>* out, std::string* error);

}  // namespace detail
}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_INFLATE_H_
