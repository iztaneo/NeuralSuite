// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/png.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/png.h"

namespace neuralsuite {
namespace image {

bool DecodePng(const uint8_t* data, size_t size, Bitmap* out, std::string* error) {
  using namespace detail;
  static const uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

  if (size < 8 || std::memcmp(data, kSignature, 8) != 0) {
    *error = "no es un archivo PNG: la firma no coincide";
    return false;
  }

  int width = 0, height = 0, depth = 0, color_type = 0, interlace = 0;
  bool have_header = false;
  std::vector<uint8_t> palette;      // RGB, 3 bytes por entrada
  std::vector<uint8_t> palette_alpha;
  std::vector<uint8_t> compressed;

  size_t pos = 8;
  while (pos + 8 <= size) {
    const uint32_t length = ReadBigEndian32(data + pos);
    // El limite de la norma es 2^31-1; ademas debe caber en lo que queda.
    if (length > 0x7FFFFFFFu || pos + 12 + length > size) {
      *error = "un trozo declara un tamano que no cabe en el archivo";
      return false;
    }
    const uint8_t* type = data + pos + 4;
    const uint8_t* body = data + pos + 8;

    const uint32_t stored_crc = ReadBigEndian32(body + length);
    if ((Crc32(type, length + 4) ^ 0xFFFFFFFFu) != stored_crc) {
      *error = "el CRC de un trozo no coincide: el archivo esta corrupto";
      return false;
    }

    if (std::memcmp(type, "IHDR", 4) == 0) {
      if (length != 13) {
        *error = "la cabecera IHDR no mide 13 bytes";
        return false;
      }
      width = static_cast<int>(ReadBigEndian32(body));
      height = static_cast<int>(ReadBigEndian32(body + 4));
      depth = body[8];
      color_type = body[9];
      interlace = body[12];
      if (width <= 0 || height <= 0) {
        *error = "la imagen declara un tamano no positivo";
        return false;
      }
      // Techo deliberado: sin el, una cabecera de trece bytes puede pedir
      // decenas de gigabytes antes de que se lea un solo pixel.
      if (static_cast<int64_t>(width) * height > 64LL * 1024 * 1024) {
        *error = "la imagen declara mas de 64 megapixeles";
        return false;
      }
      if (body[10] != 0 || body[11] != 0) {
        *error = "el archivo usa una compresion o un filtrado no normalizados";
        return false;
      }
      if (interlace != 0 && interlace != 1) {
        *error = "modo de entrelazado desconocido";
        return false;
      }
      const bool depth_ok =
          (color_type == 3) ? (depth == 1 || depth == 2 || depth == 4 || depth == 8)
                            : (color_type == 0 ? (depth == 1 || depth == 2 || depth == 4 ||
                                                  depth == 8 || depth == 16)
                                               : (depth == 8 || depth == 16));
      if ((color_type > 6 || color_type == 1 || color_type == 5) || !depth_ok) {
        *error = "combinacion de tipo de color y profundidad no valida";
        return false;
      }
      have_header = true;
    } else if (std::memcmp(type, "PLTE", 4) == 0) {
      if (length % 3 != 0 || length > 256 * 3) {
        *error = "la paleta tiene un tamano invalido";
        return false;
      }
      palette.assign(body, body + length);
    } else if (std::memcmp(type, "tRNS", 4) == 0) {
      palette_alpha.assign(body, body + length);
    } else if (std::memcmp(type, "IDAT", 4) == 0) {
      compressed.insert(compressed.end(), body, body + length);
    } else if (std::memcmp(type, "IEND", 4) == 0) {
      break;
    }
    pos += 12 + length;
  }

  if (!have_header) {
    *error = "el archivo no contiene la cabecera IHDR";
    return false;
  }
  if (compressed.empty()) {
    *error = "el archivo no contiene datos de imagen";
    return false;
  }
  if (color_type == 3 && palette.empty()) {
    *error = "la imagen es de paleta pero no trae ninguna";
    return false;
  }

  // Canales que trae el archivo, antes de expandir la paleta.
  const int raw_channels =
      (color_type == 0) ? 1 : (color_type == 2) ? 3 : (color_type == 3) ? 1 : (color_type == 4) ? 2 : 4;
  // Canales que se entregan: la paleta se expande a RGB, o a RGBA si hay tRNS.
  const int out_channels =
      (color_type == 3) ? (palette_alpha.empty() ? 3 : 4) : raw_channels;

  // Cuanto puede ocupar lo descomprimido: cada fila lleva un byte de filtro.
  const auto row_bytes = [&](int w) -> size_t {
    return (static_cast<size_t>(w) * raw_channels * depth + 7) / 8;
  };
  size_t max_output = 0;
  if (interlace == 0) {
    max_output = (row_bytes(width) + 1) * static_cast<size_t>(height);
  } else {
    for (const Adam7Pass& pass : kAdam7) {
      const int pw = (width - pass.x_start + pass.x_step - 1) / pass.x_step;
      const int ph = (height - pass.y_start + pass.y_step - 1) / pass.y_step;
      if (pw > 0 && ph > 0) max_output += (row_bytes(pw) + 1) * static_cast<size_t>(ph);
    }
  }

  std::vector<uint8_t> raw;
  if (!ZlibInflate(compressed.data(), compressed.size(), max_output, &raw, error)) return false;
  if (raw.size() != max_output) {
    *error = "los datos descomprimidos no miden lo que exige la cabecera";
    return false;
  }

  out->width = width;
  out->height = height;
  out->channels = out_channels;
  out->pixels.assign(static_cast<size_t>(width) * height * out_channels, 0);

  // Escribe un pixel ya desfiltrado en su sitio del resultado.
  const auto emit = [&](int x, int y, const uint8_t* row, size_t index) {
    uint8_t* dst = out->pixels.data() + (static_cast<size_t>(y) * width + x) * out_channels;
    if (color_type == 3) {
      const uint16_t entry = ReadSample(row, index, depth);
      if (static_cast<size_t>(entry) * 3 + 2 < palette.size()) {
        dst[0] = palette[entry * 3];
        dst[1] = palette[entry * 3 + 1];
        dst[2] = palette[entry * 3 + 2];
      }
      if (out_channels == 4) {
        dst[3] = entry < palette_alpha.size() ? palette_alpha[entry] : 255;
      }
      return;
    }
    for (int c = 0; c < raw_channels; ++c) {
      const uint16_t sample = ReadSample(row, index * raw_channels + c, depth);
      if (depth == 16) {
        dst[c] = static_cast<uint8_t>(sample >> 8);
      } else if (depth == 8) {
        dst[c] = static_cast<uint8_t>(sample);
      } else {
        // 1, 2 y 4 bits se estiran al rango completo: con 1 bit, 1 debe dar 255
        // y no 1, o la imagen saldria practicamente negra.
        const int max_value = (1 << depth) - 1;
        dst[c] = static_cast<uint8_t>(sample * 255 / max_value);
      }
    }
  };

  const size_t bpp = std::max<size_t>(1, static_cast<size_t>(raw_channels) * depth / 8);
  size_t offset = 0;

  // Un solo recorrido sirve para los dos modos: sin entrelazar es el caso de un
  // unico paso que cubre la imagen entera.
  const Adam7Pass single = {0, 0, 1, 1};
  const int n_passes = (interlace == 0) ? 1 : 7;
  for (int p = 0; p < n_passes; ++p) {
    const Adam7Pass& pass = (interlace == 0) ? single : kAdam7[p];
    const int pw = (width - pass.x_start + pass.x_step - 1) / pass.x_step;
    const int ph = (height - pass.y_start + pass.y_step - 1) / pass.y_step;
    if (pw <= 0 || ph <= 0) continue;

    const size_t stride = row_bytes(pw);
    std::vector<uint8_t> previous(stride, 0);
    std::vector<uint8_t> current(stride, 0);
    bool first_row = true;

    for (int row = 0; row < ph; ++row) {
      const uint8_t filter = raw[offset++];
      std::memcpy(current.data(), raw.data() + offset, stride);
      offset += stride;
      if (!UnfilterRow(filter, current.data(), first_row ? nullptr : previous.data(), stride, bpp,
                       error)) {
        return false;
      }
      const int y = pass.y_start + row * pass.y_step;
      for (int col = 0; col < pw; ++col) {
        emit(pass.x_start + col * pass.x_step, y, current.data(), static_cast<size_t>(col));
      }
      previous.swap(current);
      first_row = false;
    }
  }
  return true;
}

namespace detail {

uint32_t Crc32(const uint8_t* data, size_t size, uint32_t crc) {
  static const std::vector<uint32_t> table = [] {
    std::vector<uint32_t> t(256);
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      t[n] = c;
    }
    return t;
  }();
  for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
  return crc;
}

uint32_t ReadBigEndian32(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint8_t PaethPredictor(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return static_cast<uint8_t>(a);
  return pb <= pc ? static_cast<uint8_t>(b) : static_cast<uint8_t>(c);
}

bool UnfilterRow(uint8_t filter, uint8_t* row, const uint8_t* prev, size_t bytes, size_t bpp, std::string* error) {
  switch (filter) {
    case 0:
      break;
    case 1:  // Sub: el pixel de la izquierda
      for (size_t i = bpp; i < bytes; ++i) row[i] = static_cast<uint8_t>(row[i] + row[i - bpp]);
      break;
    case 2:  // Up: el pixel de arriba
      if (prev) {
        for (size_t i = 0; i < bytes; ++i) row[i] = static_cast<uint8_t>(row[i] + prev[i]);
      }
      break;
    case 3:  // Average: la media de izquierda y arriba
      for (size_t i = 0; i < bytes; ++i) {
        const int left = i >= bpp ? row[i - bpp] : 0;
        const int up = prev ? prev[i] : 0;
        row[i] = static_cast<uint8_t>(row[i] + ((left + up) >> 1));
      }
      break;
    case 4:  // Paeth
      for (size_t i = 0; i < bytes; ++i) {
        const int left = i >= bpp ? row[i - bpp] : 0;
        const int up = prev ? prev[i] : 0;
        const int corner = (i >= bpp && prev) ? prev[i - bpp] : 0;
        row[i] = static_cast<uint8_t>(row[i] + PaethPredictor(left, up, corner));
      }
      break;
    default:
      *error = "tipo de filtro desconocido en una fila: " + std::to_string(filter);
      return false;
  }
  return true;
}

uint16_t ReadSample(const uint8_t* row, size_t index, int depth) {
  if (depth == 8) return row[index];
  if (depth == 16) return static_cast<uint16_t>((row[index * 2] << 8) | row[index * 2 + 1]);
  const int per_byte = 8 / depth;
  const uint8_t byte = row[index / per_byte];
  const int shift = 8 - depth * static_cast<int>(1 + index % per_byte);
  return static_cast<uint16_t>((byte >> shift) & ((1u << depth) - 1u));
}

}  // namespace detail


}  // namespace image
}  // namespace neuralsuite
