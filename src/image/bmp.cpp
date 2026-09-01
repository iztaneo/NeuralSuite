// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/bmp.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/bmp.h"

namespace neuralsuite {
namespace image {

bool DecodeBmp(const uint8_t* data, size_t size, Bitmap* out, std::string* error) {
  using namespace detail;

  if (size < 14 || data[0] != 'B' || data[1] != 'M') {
    *error = "no es un archivo BMP: falta la firma 'BM'";
    return false;
  }
  const uint32_t pixel_offset = ReadLittleEndian32(data + 10);
  if (size < 18) {
    *error = "archivo BMP truncado en la cabecera";
    return false;
  }
  const uint32_t header_size = ReadLittleEndian32(data + 14);
  if (header_size < 12 || 14 + header_size > size) {
    *error = "la cabecera DIB declara un tamano que no cabe en el archivo";
    return false;
  }
  const uint8_t* dib = data + 14;

  int width = 0, height = 0, bpp = 0;
  uint32_t compression = 0, palette_count = 0;
  bool core_header = (header_size == 12);

  if (core_header) {
    // Cabecera de OS/2: dimensiones de 16 bits y paleta de tripletes.
    width = static_cast<int16_t>(ReadLittleEndian16(dib + 4));
    height = static_cast<int16_t>(ReadLittleEndian16(dib + 6));
    bpp = ReadLittleEndian16(dib + 10);
  } else {
    if (header_size < 40) {
      *error = "cabecera DIB de un tamano no reconocido";
      return false;
    }
    width = static_cast<int32_t>(ReadLittleEndian32(dib + 4));
    height = static_cast<int32_t>(ReadLittleEndian32(dib + 8));
    bpp = ReadLittleEndian16(dib + 14);
    compression = ReadLittleEndian32(dib + 16);
    palette_count = ReadLittleEndian32(dib + 32);
  }

  const bool top_down = height < 0;
  if (top_down) height = -height;
  if (width <= 0 || height <= 0) {
    *error = "el BMP declara un tamano no valido";
    return false;
  }
  if (static_cast<int64_t>(width) * height > 64LL * 1024 * 1024) {
    *error = "la imagen declara mas de 64 megapixeles";
    return false;
  }
  if (compression == 1 || compression == 2) {
    *error = "los BMP comprimidos con RLE no se admiten";
    return false;
  }
  if (compression != 0 && compression != 3) {
    *error = "metodo de compresion BMP desconocido: " + std::to_string(compression);
    return false;
  }
  if (bpp != 1 && bpp != 4 && bpp != 8 && bpp != 16 && bpp != 24 && bpp != 32) {
    *error = "profundidad de color no admitida en BMP: " + std::to_string(bpp);
    return false;
  }

  // Mascaras de BI_BITFIELDS. Los valores por defecto son los que la norma
  // asume cuando el archivo no las declara.
  uint32_t mask_r = 0, mask_g = 0, mask_b = 0, mask_a = 0;
  if (compression == 3) {
    if (14 + header_size + 12 > size && header_size < 52) {
      *error = "el BMP usa mascaras de bits pero no las incluye";
      return false;
    }
    const uint8_t* masks = (header_size >= 52) ? dib + 40 : data + 14 + header_size;
    mask_r = ReadLittleEndian32(masks);
    mask_g = ReadLittleEndian32(masks + 4);
    mask_b = ReadLittleEndian32(masks + 8);
    if (header_size >= 56) mask_a = ReadLittleEndian32(masks + 12);
  } else if (bpp == 16) {
    mask_r = 0x7C00; mask_g = 0x03E0; mask_b = 0x001F;  // 5-5-5
  } else if (bpp == 32) {
    mask_r = 0x00FF0000; mask_g = 0x0000FF00; mask_b = 0x000000FF;
  }

  // Paleta, solo para 1, 4 y 8 bits.
  std::vector<uint8_t> palette;
  if (bpp <= 8) {
    const size_t entry = core_header ? 3u : 4u;
    size_t entries = palette_count;
    if (entries == 0) entries = static_cast<size_t>(1) << bpp;
    const size_t table_offset = 14 + header_size;
    if (table_offset + entries * entry > size) {
      *error = "la paleta del BMP no cabe en el archivo";
      return false;
    }
    palette.resize(entries * 3);
    for (size_t i = 0; i < entries; ++i) {
      const uint8_t* src = data + table_offset + i * entry;
      palette[i * 3 + 0] = src[2];  // el archivo guarda BGR
      palette[i * 3 + 1] = src[1];
      palette[i * 3 + 2] = src[0];
    }
  }

  const size_t stride = ((static_cast<size_t>(width) * bpp + 31) / 32) * 4;
  if (pixel_offset > size || stride * static_cast<size_t>(height) > size - pixel_offset) {
    *error = "los datos de pixeles no caben en el archivo";
    return false;
  }

  const bool has_alpha = (bpp == 32 && mask_a != 0);
  out->width = width;
  out->height = height;
  out->channels = has_alpha ? 4 : 3;
  out->pixels.assign(static_cast<size_t>(width) * height * out->channels, 0);

  int shift_r = 0, width_r = 0, shift_g = 0, width_g = 0, shift_b = 0, width_b = 0;
  int shift_a = 0, width_a = 0;
  MaskShiftWidth(mask_r, &shift_r, &width_r);
  MaskShiftWidth(mask_g, &shift_g, &width_g);
  MaskShiftWidth(mask_b, &shift_b, &width_b);
  MaskShiftWidth(mask_a, &shift_a, &width_a);

  // Lleva un campo de `bits` de ancho al rango completo de 8: con 5 bits, el
  // valor maximo 31 tiene que dar 255 y no 31.
  const auto expand = [](uint32_t value, int bits) -> uint8_t {
    if (bits <= 0) return 0;
    if (bits == 8) return static_cast<uint8_t>(value);
    return static_cast<uint8_t>(value * 255u / ((1u << bits) - 1u));
  };

  for (int y = 0; y < height; ++y) {
    // El origen del formato esta abajo salvo que el alto sea negativo.
    const int src_row = top_down ? y : (height - 1 - y);
    const uint8_t* row = data + pixel_offset + static_cast<size_t>(src_row) * stride;
    uint8_t* dst = out->pixels.data() + static_cast<size_t>(y) * width * out->channels;

    for (int x = 0; x < width; ++x) {
      uint8_t* p = dst + static_cast<size_t>(x) * out->channels;
      if (bpp <= 8) {
        const int per_byte = 8 / bpp;
        const uint8_t byte = row[x / per_byte];
        const int shift = 8 - bpp * (1 + x % per_byte);
        const uint32_t index = (byte >> shift) & ((1u << bpp) - 1u);
        if (static_cast<size_t>(index) * 3 + 2 < palette.size()) {
          p[0] = palette[index * 3];
          p[1] = palette[index * 3 + 1];
          p[2] = palette[index * 3 + 2];
        }
      } else if (bpp == 24) {
        p[0] = row[x * 3 + 2];
        p[1] = row[x * 3 + 1];
        p[2] = row[x * 3 + 0];
      } else {
        const uint32_t value = (bpp == 16) ? ReadLittleEndian16(row + x * 2)
                                           : ReadLittleEndian32(row + x * 4);
        p[0] = expand((value & mask_r) >> shift_r, width_r);
        p[1] = expand((value & mask_g) >> shift_g, width_g);
        p[2] = expand((value & mask_b) >> shift_b, width_b);
        if (has_alpha) p[3] = expand((value & mask_a) >> shift_a, width_a);
      }
    }
  }
  return true;
}

namespace detail {

uint16_t ReadLittleEndian16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t ReadLittleEndian32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void MaskShiftWidth(uint32_t mask, int* shift, int* width) {
  *shift = 0;
  *width = 0;
  if (mask == 0) return;
  while (((mask >> *shift) & 1u) == 0u) ++(*shift);
  uint32_t m = mask >> *shift;
  while (m & 1u) {
    ++(*width);
    m >>= 1;
  }
}

}  // namespace detail


}  // namespace image
}  // namespace neuralsuite
