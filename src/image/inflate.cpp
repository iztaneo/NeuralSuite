// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/inflate.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/inflate.h"

namespace neuralsuite {
namespace image {

namespace detail {

void BuildFixedTables(HuffmanTable* literals, HuffmanTable* distances) {
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

bool ReadDynamicTables(BitReader& reader, HuffmanTable* literals, HuffmanTable* distances, std::string* error) {
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

bool Inflate(const uint8_t* data, size_t size, size_t max_output, std::vector<uint8_t>* out, std::string* error) {
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

uint32_t Adler32(const uint8_t* data, size_t size) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < size; ++i) {
    a = (a + data[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

bool ZlibInflate(const uint8_t* data, size_t size, size_t max_output, std::vector<uint8_t>* out, std::string* error) {
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
