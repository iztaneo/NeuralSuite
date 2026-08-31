// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/renglones.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/renglones.h"

namespace neuralsuite {
namespace image {

std::vector<Renglon> DetectarRenglones(const Bitmap& imagen, int alto_minimo, int union_maxima, float proporcion_acento, Binarizacion metodo) {
  std::vector<Renglon> renglones;
  if (imagen.Empty()) return renglones;

  std::vector<float> gris;
  ToGrayscale(imagen, &gris);
  const int ancho = imagen.width, alto = imagen.height;
  // Se da por supuesto texto oscuro sobre fondo claro, que es como llega un
  // documento.
  const std::vector<uint8_t> marca = Binarizar(gris, ancho, alto, metodo);

  std::vector<int> tinta(alto, 0);
  for (int y = 0; y < alto; ++y) {
    for (int x = 0; x < ancho; ++x) {
      if (marca[static_cast<size_t>(y) * ancho + x]) ++tinta[y];
    }
  }

  // Bandas de filas consecutivas con tinta.
  std::vector<std::pair<int, int>> bandas;
  bool dentro = false;
  int inicio = 0;
  for (int y = 0; y < alto; ++y) {
    if (tinta[y] > 0 && !dentro) {
      dentro = true;
      inicio = y;
    } else if (tinta[y] == 0 && dentro) {
      dentro = false;
      bandas.emplace_back(inicio, y - 1);
    }
  }
  if (dentro) bandas.emplace_back(inicio, alto - 1);

  // Unir acentos, tildes y puntos con el cuerpo de su letra.
  std::vector<std::pair<int, int>> unidas;
  for (const auto& banda : bandas) {
    bool unir = false;
    if (!unidas.empty()) {
      const int hueco = banda.first - unidas.back().second - 1;
      const int alto_previo = unidas.back().second - unidas.back().first + 1;
      const int alto_actual = banda.second - banda.first + 1;
      const int menor = std::min(alto_previo, alto_actual);
      const int mayor = std::max(alto_previo, alto_actual);
      unir = hueco <= union_maxima &&
             static_cast<float>(menor) <= proporcion_acento * static_cast<float>(mayor);
    }
    if (unir) {
      unidas.back().second = banda.second;
    } else {
      unidas.push_back(banda);
    }
  }

  for (const auto& banda : unidas) {
    if (banda.second - banda.first + 1 < alto_minimo) continue;

    // Recortar tambien a los lados: el renglon no suele ocupar todo el ancho, y
    // dejarle margen en blanco reduce la resolucion util al reescalarlo.
    int x0 = ancho, x1 = -1;
    for (int y = banda.first; y <= banda.second; ++y) {
      for (int x = 0; x < ancho; ++x) {
        if (marca[static_cast<size_t>(y) * ancho + x]) {
          x0 = std::min(x0, x);
          x1 = std::max(x1, x);
        }
      }
    }
    if (x1 < x0) continue;

    Renglon r;
    r.x = x0;
    r.y = banda.first;
    r.ancho = x1 - x0 + 1;
    r.alto = banda.second - banda.first + 1;
    renglones.push_back(r);
  }
  return renglones;
}

Bitmap RecortarRenglon(const Bitmap& imagen, const Renglon& r, float proporcion) {
  // Alto total que hace que r.alto sea esa proporcion, y de ahi el margen.
  const int alto_deseado = static_cast<int>(std::lround(r.alto / std::max(0.05f, proporcion)));
  const int margen_v = std::max(1, (alto_deseado - r.alto) / 2);
  // A lo ancho basta un margen pequeno: lo que importa es la proporcion
  // vertical, que es la que fija cuantos pasos abarca cada letra.
  const int margen_h = std::max(1, margen_v / 2);

  const int x0 = std::max(0, r.x - margen_h);
  const int y0 = std::max(0, r.y - margen_v);
  const int x1 = std::min(imagen.width - 1, r.x + r.ancho - 1 + margen_h);
  const int y1 = std::min(imagen.height - 1, r.y + r.alto - 1 + margen_v);

  Bitmap recorte;
  recorte.width = x1 - x0 + 1;
  recorte.height = y1 - y0 + 1;
  recorte.channels = imagen.channels;
  recorte.pixels.assign(static_cast<size_t>(recorte.width) * recorte.height * recorte.channels, 0);

  for (int y = 0; y < recorte.height; ++y) {
    const uint8_t* origen =
        imagen.pixels.data() +
        (static_cast<size_t>(y0 + y) * imagen.width + x0) * imagen.channels;
    uint8_t* destino =
        recorte.pixels.data() + static_cast<size_t>(y) * recorte.width * recorte.channels;
    std::memcpy(destino, origen,
                static_cast<size_t>(recorte.width) * recorte.channels * sizeof(uint8_t));
  }
  return recorte;
}

Tensor RenglonATensor(const Bitmap& recorte, int alto_objetivo, int multiplo_ancho, bool invertir) {
  std::vector<float> gris;
  ToGrayscale(recorte, &gris);

  int ancho = static_cast<int>(std::lround(static_cast<double>(recorte.width) * alto_objetivo /
                                           std::max(1, recorte.height)));
  ancho = std::max(multiplo_ancho, ancho);
  ancho = ((ancho + multiplo_ancho - 1) / multiplo_ancho) * multiplo_ancho;

  std::vector<float> escalado;
  Resize(gris, recorte.width, recorte.height, &escalado, ancho, alto_objetivo);

  Tensor salida({1, 1, alto_objetivo, ancho});
  for (size_t i = 0; i < escalado.size(); ++i) {
    salida[i] = invertir ? 1.0f - escalado[i] : escalado[i];
  }
  return salida;
}

}  // namespace image
}  // namespace neuralsuite
