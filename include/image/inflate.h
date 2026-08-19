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
inline void BuildFixedTables(HuffmanTable* literals, HuffmanTable* distances) {
  uint8_t lengths[288];
  for (int i = 0; i < 144; ++i) lengths[i] = 8;
  for (int i = 144; i < 256; ++i) lengths[i] = 9;
  for (int i = 256; i < 280; ++i) lengths[i] = 7;
  for (int i = 280; i < 288; ++i) lengths[i] = 8;
  literals->Build(lengths, 288);

  uint8_t dist_lengths[30];
  for (int i = 0; i < 30; ++i) dist_lengths[i] = 5;
  distances->Build(dist_lengths, 30);
}

/**
 * @brief Lee el arbol propio de un bloque de tipo 10.
 *
 * Las longitudes de codigo se transmiten a su vez comprimidas con un tercer
 * arbol, cuyas longitudes vienen en un orden fijo pensado para que las mas
 * probables queden delante y las ultimas puedan omitirse.
 */
inline bool ReadDynamicTables(BitReader& reader, HuffmanTable* literals,
                              HuffmanTable* distances, std::string* error) {
  static const uint8_t kOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                     11, 4,  12, 3, 13, 2, 14, 1, 15};
  uint32_t hlit = 0, hdist = 0, hclen = 0;
  if (!reader.Bits(5, &hlit) || !reader.Bits(5, &hdist) || !reader.Bits(4, &hclen)) {
    *error = "flujo truncado en la cabecera del bloque dinamico";
    return false;
  }
  const int n_lit = static_cast<int>(hlit) + 257;
  const int n_dist = static_cast<int>(hdist) + 1;
  const int n_clen = static_cast<int>(hclen) + 4;
  if (n_lit > 286 || n_dist > 30) {
    *error = "el bloque dinamico declara mas codigos de los que permite la norma";
    return false;
  }

  uint8_t clen[19] = {0};
  for (int i = 0; i < n_clen; ++i) {
    uint32_t value = 0;
    if (!reader.Bits(3, &value)) {
      *error = "flujo truncado leyendo las longitudes del arbol auxiliar";
      return false;
    }
    clen[kOrder[i]] = static_cast<uint8_t>(value);
  }
  HuffmanTable code_lengths;
  if (!code_lengths.Build(clen, 19)) {
    *error = "el arbol auxiliar de longitudes no es valido";
    return false;
  }

  std::vector<uint8_t> lengths(static_cast<size_t>(n_lit + n_dist), 0);
  size_t i = 0;
  while (i < lengths.size()) {
    const int symbol = code_lengths.Decode(reader);
    if (symbol < 0) {
      *error = "codigo invalido en las longitudes del bloque dinamico";
      return false;
    }
    if (symbol < 16) {
      lengths[i++] = static_cast<uint8_t>(symbol);
      continue;
    }
    // 16 repite la longitud anterior; 17 y 18 insertan rachas de ceros.
    int repeat = 0;
    uint8_t value = 0;
    uint32_t extra = 0;
    if (symbol == 16) {
      if (i == 0) {
        *error = "la primera longitud del bloque pide repetir la anterior";
        return false;
      }
      if (!reader.Bits(2, &extra)) { *error = "flujo truncado en una repeticion"; return false; }
      repeat = 3 + static_cast<int>(extra);
      value = lengths[i - 1];
    } else if (symbol == 17) {
      if (!reader.Bits(3, &extra)) { *error = "flujo truncado en una racha de ceros"; return false; }
      repeat = 3 + static_cast<int>(extra);
    } else {
      if (!reader.Bits(7, &extra)) { *error = "flujo truncado en una racha de ceros"; return false; }
      repeat = 11 + static_cast<int>(extra);
    }
    if (i + static_cast<size_t>(repeat) > lengths.size()) {
      *error = "una repeticion de longitudes se sale de la tabla";
      return false;
    }
    for (int r = 0; r < repeat; ++r) lengths[i++] = value;
  }

  if (lengths[256] == 0) {
    *error = "el bloque dinamico no define el codigo de fin de bloque";
    return false;
  }
  if (!literals->Build(lengths.data(), n_lit) ||
      !distances->Build(lengths.data() + n_lit, n_dist)) {
    *error = "los arboles del bloque dinamico no son validos";
    return false;
  }
  return true;
}

/**
 * @brief Descomprime un flujo DEFLATE crudo, sin la cabecera de zlib.
 *
 * `max_output` acota cuanto se puede llegar a producir: sin ese limite, un
 * archivo pequeno y malicioso puede pedir gigabytes de memoria.
 */
inline bool Inflate(const uint8_t* data, size_t size, size_t max_output,
                    std::vector<uint8_t>* out, std::string* error) {
  BitReader reader(data, size);
  out->clear();

  for (;;) {
    uint32_t final_block = 0, type = 0;
    if (!reader.Bits(1, &final_block) || !reader.Bits(2, &type)) {
      *error = "flujo truncado en la cabecera de un bloque";
      return false;
    }

    if (type == 0) {
      // Bloque sin comprimir: longitud y su complemento, como comprobacion.
      reader.AlignToByte();
      uint8_t header[4];
      if (!reader.ReadBytes(header, 4)) {
        *error = "flujo truncado en un bloque sin comprimir";
        return false;
      }
      const uint16_t len = static_cast<uint16_t>(header[0] | (header[1] << 8));
      const uint16_t nlen = static_cast<uint16_t>(header[2] | (header[3] << 8));
      if (static_cast<uint16_t>(~len & 0xFFFF) != nlen) {
        *error = "la longitud de un bloque sin comprimir no cuadra con su complemento";
        return false;
      }
      if (out->size() + len > max_output) {
        *error = "la salida descomprimida supera el tamano esperado";
        return false;
      }
      const size_t offset = out->size();
      out->resize(offset + len);
      if (!reader.ReadBytes(out->data() + offset, len)) {
        *error = "flujo truncado dentro de un bloque sin comprimir";
        return false;
      }
    } else if (type == 1 || type == 2) {
      HuffmanTable literals, distances;
      if (type == 1) {
        BuildFixedTables(&literals, &distances);
      } else if (!ReadDynamicTables(reader, &literals, &distances, error)) {
        return false;
      }

      for (;;) {
        const int symbol = literals.Decode(reader);
        if (symbol < 0) {
          *error = "codigo invalido en el cuerpo de un bloque";
          return false;
        }
        if (symbol == 256) break;  // fin de bloque

        if (symbol < 256) {
          if (out->size() + 1 > max_output) {
            *error = "la salida descomprimida supera el tamano esperado";
            return false;
          }
          out->push_back(static_cast<uint8_t>(symbol));
          continue;
        }

        // Referencia hacia atras: longitud y distancia.
        const int length_index = symbol - 257;
        if (length_index >= 29) {
          *error = "simbolo de longitud fuera de rango";
          return false;
        }
        uint32_t extra = 0;
        if (!reader.Bits(kLengthExtra[length_index], &extra)) {
          *error = "flujo truncado leyendo una longitud";
          return false;
        }
        const size_t length = kLengthBase[length_index] + extra;

        const int dist_symbol = distances.Decode(reader);
        if (dist_symbol < 0 || dist_symbol >= 30) {
          *error = "simbolo de distancia invalido";
          return false;
        }
        if (!reader.Bits(kDistExtra[dist_symbol], &extra)) {
          *error = "flujo truncado leyendo una distancia";
          return false;
        }
        const size_t distance = kDistBase[dist_symbol] + extra;
        if (distance > out->size()) {
          *error = "una referencia apunta antes del inicio de la salida";
          return false;
        }
        if (out->size() + length > max_output) {
          *error = "la salida descomprimida supera el tamano esperado";
          return false;
        }

        // La copia es byte a byte a proposito: los tramos pueden solaparse
        // -distancia 1 y longitud 100 repite el ultimo byte cien veces-, asi que
        // memcpy daria un resultado distinto.
        size_t from = out->size() - distance;
        for (size_t k = 0; k < length; ++k) out->push_back((*out)[from + k]);
      }
    } else {
      *error = "tipo de bloque reservado";
      return false;
    }

    if (final_block) break;
  }
  return true;
}

/** @brief Suma de comprobacion Adler-32, la que lleva la envoltura zlib. */
inline uint32_t Adler32(const uint8_t* data, size_t size) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < size; ++i) {
    a = (a + data[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

/**
 * @brief Descomprime un flujo zlib: cabecera de dos bytes, DEFLATE y Adler-32.
 *
 * La suma se comprueba de verdad. Aceptarla sin mirar dejaria pasar en silencio
 * un archivo corrupto, que es como se cuelan los defectos dificiles de ubicar:
 * el error aparece mucho despues, en el modelo, y parece cosa del modelo.
 */
inline bool ZlibInflate(const uint8_t* data, size_t size, size_t max_output,
                        std::vector<uint8_t>* out, std::string* error) {
  if (size < 6) {
    *error = "flujo zlib demasiado corto";
    return false;
  }
  const uint8_t cmf = data[0], flg = data[1];
  if ((cmf & 0x0F) != 8) {
    *error = "el flujo zlib no usa el metodo DEFLATE";
    return false;
  }
  if ((static_cast<uint32_t>(cmf) * 256u + flg) % 31u != 0u) {
    *error = "la cabecera zlib no pasa su propia comprobacion";
    return false;
  }
  if (flg & 0x20) {
    *error = "el flujo zlib usa un diccionario predefinido, que no se admite";
    return false;
  }

  if (!Inflate(data + 2, size - 2, max_output, out, error)) return false;

  const uint32_t expected = (static_cast<uint32_t>(data[size - 4]) << 24) |
                            (static_cast<uint32_t>(data[size - 3]) << 16) |
                            (static_cast<uint32_t>(data[size - 2]) << 8) |
                            static_cast<uint32_t>(data[size - 1]);
  if (Adler32(out->data(), out->size()) != expected) {
    *error = "la suma Adler-32 no coincide: el flujo comprimido esta corrupto";
    return false;
  }
  return true;
}

}  // namespace detail
}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_INFLATE_H_
