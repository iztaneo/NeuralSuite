// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/jpeg.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "tensor.h"
#include "image/jpeg.h"

namespace neuralsuite {
namespace image {

bool DecodeJpeg(const uint8_t* data, size_t size, Bitmap* out, std::string* error) {
  using namespace jpeg_detail;

  if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
    *error = "no es un archivo JPEG: falta el marcador de inicio";
    return false;
  }

  uint16_t quant[4][64] = {};
  bool quant_defined[4] = {};
  HuffmanTable dc_tables[4], ac_tables[4];
  std::vector<Component> components;
  int frame_width = 0, frame_height = 0;
  int h_max = 1, v_max = 1;
  int mcus_per_line = 0, mcus_per_column = 0;
  bool progressive = false;
  bool have_frame = false;
  int restart_interval = 0;
  // El marcador APP14 de Adobe indica si un archivo de tres componentes esta
  // realmente en YCbCr o directamente en RGB, cosa que no se puede deducir.
  int adobe_transform = -1;

  size_t pos = 2;
  const auto read16 = [&](size_t at) -> int {
    return (static_cast<int>(data[at]) << 8) | data[at + 1];
  };

  while (pos + 1 < size) {
    if (data[pos] != 0xFF) { ++pos; continue; }
    while (pos < size && data[pos] == 0xFF) ++pos;   // el relleno de 0xFF es legal
    if (pos >= size) break;
    const uint8_t marker = data[pos++];

    if (marker == 0xD9) break;                        // EOI
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;  // sin cuerpo

    if (pos + 2 > size) { *error = "archivo truncado en la cabecera de un segmento"; return false; }
    const int length = read16(pos);
    if (length < 2 || pos + length > size) {
      *error = "un segmento declara un tamano que no cabe en el archivo";
      return false;
    }
    const uint8_t* body = data + pos + 2;
    const int body_length = length - 2;

    switch (marker) {
      case 0xC0: case 0xC1: case 0xC2: {  // SOF0 linea base, SOF1 extendido, SOF2 progresivo
        if (have_frame) { *error = "el archivo declara mas de una imagen"; return false; }
        progressive = (marker == 0xC2);
        if (body_length < 6) { *error = "cabecera de imagen incompleta"; return false; }
        if (body[0] != 8) {
          *error = "solo se admiten JPEG de 8 bits por muestra, este declara " +
                   std::to_string(body[0]);
          return false;
        }
        frame_height = (body[1] << 8) | body[2];
        frame_width = (body[3] << 8) | body[4];
        const int n = body[5];
        if (frame_width <= 0 || frame_height <= 0) {
          *error = "la imagen declara un tamano no valido";
          return false;
        }
        if (static_cast<int64_t>(frame_width) * frame_height > 64LL * 1024 * 1024) {
          *error = "la imagen declara mas de 64 megapixeles";
          return false;
        }
        if (n != 1 && n != 3) {
          *error = "solo se admiten JPEG de 1 o 3 componentes, este tiene " + std::to_string(n);
          return false;
        }
        if (body_length < 6 + 3 * n) { *error = "cabecera de imagen incompleta"; return false; }

        components.clear();
        for (int i = 0; i < n; ++i) {
          Component c;
          c.id = body[6 + i * 3];
          c.h = body[7 + i * 3] >> 4;
          c.v = body[7 + i * 3] & 15;
          c.quant_table = body[8 + i * 3];
          if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.quant_table > 3) {
            *error = "un componente declara factores de muestreo no validos";
            return false;
          }
          components.push_back(c);
          h_max = std::max(h_max, c.h);
          v_max = std::max(v_max, c.v);
        }

        mcus_per_line = (frame_width + 8 * h_max - 1) / (8 * h_max);
        mcus_per_column = (frame_height + 8 * v_max - 1) / (8 * v_max);
        for (Component& c : components) {
          c.blocks_per_line = (frame_width * c.h + 8 * h_max - 1) / (8 * h_max);
          c.blocks_per_column = (frame_height * c.v + 8 * v_max - 1) / (8 * v_max);
          // La rejilla se reserva completa hasta el borde del ultimo MCU: los
          // bloques sobrantes se codifican igual y hay que tener sitio.
          c.stride_blocks = mcus_per_line * c.h;
          c.rows_blocks = mcus_per_column * c.v;
          c.coefficients.assign(static_cast<size_t>(c.stride_blocks) * c.rows_blocks * 64, 0);
        }
        have_frame = true;
        break;
      }

      case 0xC3: case 0xC5: case 0xC6: case 0xC7:
      case 0xC9: case 0xCA: case 0xCB:
      case 0xCD: case 0xCE: case 0xCF: {
        *error = "modo JPEG no admitido (sin perdida, jerarquico o aritmetico)";
        return false;
      }

      case 0xC4: {  // DHT
        int at = 0;
        while (at < body_length) {
          if (at + 17 > body_length) { *error = "tabla Huffman incompleta"; return false; }
          const int cls = body[at] >> 4;
          const int id = body[at] & 15;
          if (cls > 1 || id > 3) { *error = "identificador de tabla Huffman no valido"; return false; }
          uint8_t counts[16];
          size_t total = 0;
          for (int i = 0; i < 16; ++i) {
            counts[i] = body[at + 1 + i];
            total += counts[i];
          }
          if (total > 256 || at + 17 + static_cast<int>(total) > body_length) {
            *error = "tabla Huffman con mas simbolos de los que caben";
            return false;
          }
          HuffmanTable& table = (cls == 0) ? dc_tables[id] : ac_tables[id];
          if (!table.Build(counts, body + at + 17, total)) {
            *error = "tabla Huffman mal formada";
            return false;
          }
          at += 17 + static_cast<int>(total);
        }
        break;
      }

      case 0xDB: {  // DQT
        int at = 0;
        while (at < body_length) {
          const int precision = body[at] >> 4;
          const int id = body[at] & 15;
          if (id > 3 || precision > 1) { *error = "tabla de cuantizacion no valida"; return false; }
          const int bytes = precision ? 128 : 64;
          if (at + 1 + bytes > body_length) {
            *error = "tabla de cuantizacion incompleta";
            return false;
          }
          for (int i = 0; i < 64; ++i) {
            quant[id][i] = precision ? static_cast<uint16_t>((body[at + 1 + i * 2] << 8) |
                                                             body[at + 2 + i * 2])
                                     : body[at + 1 + i];
          }
          quant_defined[id] = true;
          at += 1 + bytes;
        }
        break;
      }

      case 0xDD: {  // DRI
        if (body_length < 2) { *error = "intervalo de reinicio incompleto"; return false; }
        restart_interval = (body[0] << 8) | body[1];
        break;
      }

      case 0xEE: {  // APP14, el de Adobe
        if (body_length >= 12 && std::memcmp(body, "Adobe", 5) == 0) {
          adobe_transform = body[body_length - 1];
        }
        break;
      }

      case 0xDA: {  // SOS: aqui empieza un segmento de datos comprimidos
        if (!have_frame) { *error = "hay datos de imagen antes de su cabecera"; return false; }
        if (body_length < 1) { *error = "cabecera de scan incompleta"; return false; }
        const int n_scan = body[0];
        if (n_scan < 1 || n_scan > static_cast<int>(components.size()) ||
            body_length < 1 + 2 * n_scan + 3) {
          *error = "cabecera de scan incompleta";
          return false;
        }

        std::vector<Component*> scan_components;
        for (int i = 0; i < n_scan; ++i) {
          const int id = body[1 + i * 2];
          const int tables = body[2 + i * 2];
          Component* found = nullptr;
          for (Component& c : components) {
            if (c.id == id) found = &c;
          }
          if (!found) { *error = "el scan menciona un componente inexistente"; return false; }
          found->dc_table = tables >> 4;
          found->ac_table = tables & 15;
          if (found->dc_table > 3 || found->ac_table > 3) {
            *error = "el scan usa una tabla Huffman no valida";
            return false;
          }
          scan_components.push_back(found);
        }

        const int ss = body[1 + 2 * n_scan];
        const int se = body[2 + 2 * n_scan];
        const int ah = body[3 + 2 * n_scan] >> 4;
        const int al = body[3 + 2 * n_scan] & 15;
        if (ss > 63 || se > 63 || ss > se || ah > 13 || al > 13) {
          *error = "el scan declara parametros fuera de rango";
          return false;
        }

        const size_t scan_start = pos + length;
        EntropyReader reader(data + scan_start, size - scan_start);
        for (Component* c : scan_components) c->dc_pred = 0;
        int eob_run = 0;

        // Decodifica un bloque segun el modo y la fase en que estemos. Los
        // cuatro casos del progresivo estan aqui juntos porque comparten el
        // recorrido de los bloques y solo cambian en lo que escriben.
        const auto decode_block = [&](Component& c, int32_t* block) -> bool {
          if (!progressive) {
            const int t = reader.Decode(dc_tables[c.dc_table]);
            if (t < 0 || t > 15) return false;
            c.dc_pred += EntropyReader::Extend(reader.Receive(t), t);
            block[0] = c.dc_pred;

            int k = 1;
            while (k < 64) {
              const int rs = reader.Decode(ac_tables[c.ac_table]);
              if (rs < 0) return false;
              const int r = rs >> 4, s = rs & 15;
              if (s == 0) {
                if (r != 15) break;   // fin de bloque
                k += 16;              // racha de dieciseis ceros
                continue;
              }
              k += r;
              if (k > 63) break;
              block[kZigZag[k]] = EntropyReader::Extend(reader.Receive(s), s);
              ++k;
            }
            return true;
          }

          if (ss == 0) {
            // Coeficiente continuo. En la primera pasada llega su parte alta;
            // en las siguientes, un bit mas de precision.
            if (ah == 0) {
              const int t = reader.Decode(dc_tables[c.dc_table]);
              if (t < 0 || t > 15) return false;
              c.dc_pred += EntropyReader::Extend(reader.Receive(t), t);
              block[0] = c.dc_pred * (1 << al);
            } else if (reader.ReadBit()) {
              block[0] |= (1 << al);
            }
            return true;
          }

          if (ah == 0) {
            // Primera pasada de un rango de frecuencias.
            if (eob_run > 0) { --eob_run; return true; }
            int k = ss;
            while (k <= se) {
              const int rs = reader.Decode(ac_tables[c.ac_table]);
              if (rs < 0) return false;
              const int r = rs >> 4, s = rs & 15;
              if (s == 0) {
                if (r < 15) {
                  // Un fin de bloque puede abarcar varios bloques seguidos.
                  eob_run = (1 << r) - 1;
                  if (r) eob_run += reader.Receive(r);
                  break;
                }
                k += 16;
                continue;
              }
              k += r;
              if (k > se) break;
              block[kZigZag[k]] = EntropyReader::Extend(reader.Receive(s), s) * (1 << al);
              ++k;
            }
            return true;
          }

          // Refinamiento. Es la parte mas delicada de todo el formato, porque
          // el flujo lleva dos cosas entrelazadas: los bits de correccion de
          // los coeficientes que ya existen y las rachas que localizan los que
          // aparecen ahora. La regla que lo ordena todo es que **la racha solo
          // cuenta los coeficientes que hoy valen cero**: los que ya tienen
          // valor se refinan al pasar por encima y no consumen racha.
          const int positive = 1 << al;
          const int negative = -(1 << al);
          int k = ss;

          // Refina el tramo [k, se], que es lo que toca cuando un fin de bloque
          // anterior cubre este.
          const auto refine_rest = [&](int from) {
            for (int i = from; i <= se; ++i) {
              int32_t& coef = block[kZigZag[i]];
              if (coef != 0 && reader.ReadBit() && (coef & positive) == 0) {
                coef += (coef >= 0) ? positive : negative;
              }
            }
          };

          if (eob_run > 0) {
            --eob_run;
            refine_rest(k);
            return true;
          }

          bool hit_eob = false;
          for (; k <= se; ++k) {
            const int rs = reader.Decode(ac_tables[c.ac_table]);
            if (rs < 0) return false;
            int r = rs >> 4;
            const int s = rs & 15;
            int value = 0;

            if (s != 0) {
              // En refinamiento la magnitud solo puede ser 1: un bit da el signo.
              value = reader.ReadBit() ? positive : negative;
            } else if (r != 15) {
              eob_run = (1 << r) - 1;
              if (r) eob_run += reader.Receive(r);
              hit_eob = true;
              break;
            }
            // r == 15 con s == 0 salta dieciseis ceros y no inserta nada.

            while (k <= se) {
              int32_t& coef = block[kZigZag[k]];
              if (coef != 0) {
                if (reader.ReadBit() && (coef & positive) == 0) {
                  coef += (coef >= 0) ? positive : negative;
                }
              } else if (--r < 0) {
                // Aqui se para: `k` senala el hueco donde va el coeficiente
                // nuevo, y no debe avanzar. Incrementarlo aqui lo colocaba una
                // posicion mas alla y descuadraba el resto del scan.
                break;
              }
              ++k;
            }

            if (value != 0 && k <= se) block[kZigZag[k]] = value;
          }

          // Al terminar por fin de bloque quedan coeficientes por refinar en
          // este mismo bloque. Se hace siempre que se haya llegado ahi, no solo
          // cuando queden bloques por cubrir: un fin de bloque con racha de uno
          // deja el contador a cero y aun asi hay que gastar los bits de
          // correccion de este. Condicionarlo al contador desalineaba el flujo
          // a partir de ese punto.
          if (hit_eob) refine_rest(k);
          return true;
        };

        const bool interleaved = scan_components.size() > 1;
        const int rows = interleaved ? mcus_per_column : scan_components[0]->blocks_per_column;
        const int cols = interleaved ? mcus_per_line : scan_components[0]->blocks_per_line;
        const int total_units = rows * cols;
        bool failed = false;

        for (int unit = 0; unit < total_units && !failed; ++unit) {
          if (restart_interval && unit > 0 && unit % restart_interval == 0) {
            // Los marcadores de reinicio permiten recuperarse de un error: el
            // predictor vuelve a cero y los bits sueltos se descartan.
            size_t at = scan_start + reader.Position();
            while (at + 1 < size && !(data[at] == 0xFF && data[at + 1] >= 0xD0 &&
                                      data[at + 1] <= 0xD7)) {
              ++at;
            }
            if (at + 1 >= size) break;
            reader.RestartAt(at + 2 - scan_start);
            for (Component* c : scan_components) c->dc_pred = 0;
            eob_run = 0;
          }

          const int row = unit / cols, col = unit % cols;
          if (interleaved) {
            for (Component* c : scan_components) {
              for (int by = 0; by < c->v && !failed; ++by) {
                for (int bx = 0; bx < c->h && !failed; ++bx) {
                  const int block_row = row * c->v + by;
                  const int block_col = col * c->h + bx;
                  int32_t* block = c->coefficients.data() +
                                   (static_cast<size_t>(block_row) * c->stride_blocks + block_col) * 64;
                  failed = !decode_block(*c, block);
                }
              }
            }
          } else {
            Component& c = *scan_components[0];
            int32_t* block =
                c.coefficients.data() + (static_cast<size_t>(row) * c.stride_blocks + col) * 64;
            failed = !decode_block(c, block);
          }
        }

        if (failed && !reader.Exhausted()) {
          *error = "codigo Huffman invalido en los datos de imagen";
          return false;
        }

        // Saltar hasta el siguiente marcador que no sea de reinicio. El lector
        // se detiene *sobre* el codigo del marcador, con el 0xFF ya consumido,
        // asi que hay que retroceder ese byte: buscar hacia delante desde aqui
        // se saltaria el marcador que acaba de aparecer y con el, en un JPEG
        // progresivo, todo un scan.
        size_t at = scan_start + reader.Position();
        if (at > scan_start && data[at - 1] == 0xFF) --at;
        while (at + 1 < size) {
          if (data[at] == 0xFF && data[at + 1] != 0x00 &&
              !(data[at + 1] >= 0xD0 && data[at + 1] <= 0xD7)) {
            break;
          }
          ++at;
        }
        pos = at;
        continue;  // `pos` ya apunta al marcador siguiente
      }

      default:
        break;  // APPn, COM y cualquier otro segmento con cuerpo se ignoran
    }
    pos += length;
  }

  if (!have_frame) {
    *error = "el archivo no contiene ninguna imagen";
    return false;
  }

  // Cuantizacion inversa y transformada, componente a componente.
  for (Component& c : components) {
    if (!quant_defined[c.quant_table]) {
      *error = "un componente usa una tabla de cuantizacion que no se ha definido";
      return false;
    }
    const uint16_t* q = quant[c.quant_table];
    const int plane_width = c.stride_blocks * 8;
    c.samples.assign(static_cast<size_t>(plane_width) * c.rows_blocks * 8, 0);

    int32_t block[64];
    for (int by = 0; by < c.rows_blocks; ++by) {
      for (int bx = 0; bx < c.stride_blocks; ++bx) {
        const int32_t* src =
            c.coefficients.data() + (static_cast<size_t>(by) * c.stride_blocks + bx) * 64;
        // La tabla viene en orden zigzag, igual que los coeficientes al leerlos.
        for (int k = 0; k < 64; ++k) block[kZigZag[k]] = src[kZigZag[k]] * q[k];
        InverseDct(block, c.samples.data() +
                              (static_cast<size_t>(by) * 8 * plane_width + bx * 8),
                   plane_width);
      }
    }
  }

  const int channels = (components.size() == 1) ? 1 : 3;
  out->width = frame_width;
  out->height = frame_height;
  out->channels = channels;
  out->pixels.assign(static_cast<size_t>(frame_width) * frame_height * channels, 0);

  // Interpolacion de los componentes submuestreados y paso a RGB.
  //
  // La crominancia suele venir a la mitad de resolucion, asi que hay que
  // reconstruirla. Repetir el pixel mas proximo es lo mas simple y se nota: dos
  // pixeles de cada cuatro reciben un color que no les toca y aparecen bloques
  // de 2x2 en los bordes de color. Medido contra libjpeg sobre una imagen 4:2:0,
  // la diferencia media pasa de 3.7 a 0.1 al interpolar.
  //
  // La interpolacion es lineal entre centros de pixel, no entre esquinas. Con
  // esquinas la crominancia se desplaza medio pixel respecto de la luminancia,
  // que es un desalineado pequeno pero sistematico y tine los bordes. En el
  // factor 2 habitual, esta formula da los pesos 3/4 y 1/4 del filtro
  // triangular que usan las implementaciones al uso.
  //
  // Para el componente que ya viene a resolucion completa el calculo es la
  // identidad exacta: la coordenada cae justo sobre un pixel y el peso es cero.
  const auto sample_at = [&](const Component& c, int x, int y) -> double {
    const int plane_width = c.stride_blocks * 8;
    // Extension util del componente. Mas alla hay relleno que anadio el
    // codificador para completar el ultimo bloque, y no debe entrar.
    const int comp_w = (frame_width * c.h + h_max - 1) / h_max;
    const int comp_h = (frame_height * c.v + v_max - 1) / v_max;

    const double fx = (x + 0.5) * c.h / h_max - 0.5;
    const double fy = (y + 0.5) * c.v / v_max - 0.5;
    const int x0 = std::max(0, std::min(comp_w - 1, static_cast<int>(std::floor(fx))));
    const int y0 = std::max(0, std::min(comp_h - 1, static_cast<int>(std::floor(fy))));
    const int x1 = std::min(comp_w - 1, x0 + 1);
    const int y1 = std::min(comp_h - 1, y0 + 1);
    const double wx = std::max(0.0, std::min(1.0, fx - x0));
    const double wy = std::max(0.0, std::min(1.0, fy - y0));

    const uint8_t* p = c.samples.data();
    const double a = p[static_cast<size_t>(y0) * plane_width + x0];
    const double b = p[static_cast<size_t>(y0) * plane_width + x1];
    const double cc = p[static_cast<size_t>(y1) * plane_width + x0];
    const double dd = p[static_cast<size_t>(y1) * plane_width + x1];
    return a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + cc * (1 - wx) * wy + dd * wx * wy;
  };

  const auto clamp_byte = [](double v) -> uint8_t {
    const long n = std::lround(v);
    return static_cast<uint8_t>(n < 0 ? 0 : (n > 255 ? 255 : n));
  };

  for (int y = 0; y < frame_height; ++y) {
    for (int x = 0; x < frame_width; ++x) {
      uint8_t* dst = out->pixels.data() + (static_cast<size_t>(y) * frame_width + x) * channels;
      if (channels == 1) {
        dst[0] = clamp_byte(sample_at(components[0], x, y));
        continue;
      }
      const double c0 = sample_at(components[0], x, y);
      const double c1 = sample_at(components[1], x, y);
      const double c2 = sample_at(components[2], x, y);

      // adobe_transform == 0 significa que los tres canales ya son RGB.
      if (adobe_transform == 0) {
        dst[0] = clamp_byte(c0);
        dst[1] = clamp_byte(c1);
        dst[2] = clamp_byte(c2);
        continue;
      }
      const double cb = c1 - 128.0, cr = c2 - 128.0;
      dst[0] = clamp_byte(c0 + 1.402 * cr);
      dst[1] = clamp_byte(c0 - 0.344136 * cb - 0.714136 * cr);
      dst[2] = clamp_byte(c0 + 1.772 * cb);
    }
  }
  return true;
}

namespace jpeg_detail {

void InverseDct(const int32_t* input, uint8_t* output, int stride) {
  static const std::array<double, 64> kCos = [] {
    std::array<double, 64> table{};
    for (int x = 0; x < 8; ++x) {
      for (int u = 0; u < 8; ++u) {
        const double scale = (u == 0) ? std::sqrt(0.125) : 0.5;
        table[x * 8 + u] = scale * std::cos((2 * x + 1) * u * kPi / 16.0);
      }
    }
    return table;
  }();

  double intermediate[64];
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      double sum = 0.0;
      for (int u = 0; u < 8; ++u) sum += kCos[x * 8 + u] * input[y * 8 + u];
      intermediate[y * 8 + x] = sum;
    }
  }
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      double sum = 0.0;
      for (int v = 0; v < 8; ++v) sum += kCos[y * 8 + v] * intermediate[v * 8 + x];
      // Las muestras se guardan centradas en cero; hay que devolverlas al rango
      // de 0 a 255 y recortar, porque la cuantizacion puede sacarlas de el.
      const long value = std::lround(sum) + 128;
      output[y * stride + x] = static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }
  }
}

}  // namespace jpeg_detail


}  // namespace image
}  // namespace neuralsuite
