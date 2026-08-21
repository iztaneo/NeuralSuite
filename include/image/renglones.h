// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file renglones.h
 * @brief Separa una imagen de texto en renglones.
 *
 * Un CRNN reconoce **una linea**. Al darle una pagina entera, el reescalado la
 * reduce a una miniatura y devuelve basura: 432x462 pixeles acaban en 32x32.
 * Falta el paso previo, que es este.
 *
 * El metodo es la proyeccion horizontal: se cuenta cuanta tinta hay en cada
 * fila y se buscan las bandas de filas con tinta separadas por filas vacias.
 * Es lo mas simple que funciona en texto impreso, y no lleva ningun numero
 * ajustado a mano: el umbral entre tinta y fondo sale del metodo de Otsu, que
 * lo deduce del histograma de la propia imagen.
 *
 * Se comprobo sobre las dos imagenes del repositorio antes de escribirlo:
 * encuentra los 19 renglones de la pagina de la Iliada, de 11 a 14 pixeles de
 * alto, y 4 bandas en el logotipo de Mitsubishi —dos del dibujo, de 187 y 94
 * pixeles, y dos del texto, de 59—.
 *
 * Y ahi esta el limite, que conviene decir claro: **esto separa bandas, no
 * distingue texto de dibujo**. Las dos bandas del logotipo son tan renglon
 * para este codigo como las palabras de debajo. Filtrar por altura resolveria
 * ese caso concreto y seria ajustar el codigo a una imagen; la salida general
 * es que decida el reconocedor por su confianza, lo que exige entrenarlo con
 * ejemplos negativos. Eso esta anotado en el roadmap y no se hace aqui.
 *
 * Tampoco sirve para texto a varias columnas, ni girado, ni manuscrito. Para
 * eso hace falta un detector de texto de verdad, que es otro modelo.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_RENGLONES_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_RENGLONES_H_

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstring>
#include "../tensor.h"
#include "binarizar.h"
#include "bitmap.h"

namespace neuralsuite {
namespace image {

/** @brief Rectangulo de un renglon dentro de la imagen. */
struct Renglon {
  int x = 0, y = 0, ancho = 0, alto = 0;
};

/**
 * @brief Encuentra los renglones de una imagen.
 *
 * `alto_minimo` descarta bandas de una o dos filas, que en un escaneo son
 * motas y no texto.
 *
 * `union_maxima` y `proporcion_acento` vuelven a juntar los acentos y los
 * puntos de la i, que quedan despegados del cuerpo de la letra y sin esto
 * saldrian como renglones propios. Hacen falta las dos condiciones, y la
 * segunda es la que importa.
 *
 * Al principio bastaba con el hueco: unir dos bandas separadas por dos filas o
 * menos. Funcionaba en la pagina de la Iliada, donde los renglones miden 14
 * pixeles... y fallaba en un abecedario manuscrito, donde miden 75 y dos
 * renglones seguidos quedaban a dos filas uno de otro. Se fusionaban en uno de
 * 159 pixeles. El umbral en pixeles absolutos estaba ajustado, sin quererlo, a
 * la escala de una imagen concreta.
 *
 * Hacerlo relativo al alto tampoco sirve: dos pixeles sobre setenta y cinco es
 * una proporcion aun mas pequena. Lo que separa un acento de un renglon no es
 * la distancia, es el tamano: **un acento es una marca pequena junto a un
 * cuerpo grande**, y dos renglones seguidos son dos bandas de altura parecida.
 * Por eso solo se unen si una de las dos es mucho mas baja que la otra.
 */
inline std::vector<Renglon> DetectarRenglones(const Bitmap& imagen, int alto_minimo = 4,
                                              int union_maxima = 2,
                                              float proporcion_acento = 0.4f,
                                              Binarizacion metodo = Binarizacion::kAutomatica) {
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

/**
 * @brief Proporcion del alto que ocupa la tinta en las imagenes de entrenamiento.
 *
 * Medido sobre 800 imagenes del corpus: mediana 0.72, con los percentiles 10 y
 * 90 en 0.53 y 0.81. No es un numero elegido, es el que produce el generador al
 * ajustar cada palabra a su caja.
 */
constexpr float kProporcionTintaEntrenamiento = 0.72f;

/**
 * @brief Extrae un renglon como imagen propia, con margen alrededor.
 *
 * El margen no es cosmetico. Recortar al ras deja la tinta ocupando el 100% del
 * alto, y el modelo se entreno con palabras que ocupaban el 72%: al reescalar,
 * cada letra sale proporcionalmente mas ancha y abarca mas pasos de la
 * secuencia de los que vio nunca. Se noto leyendo el logotipo de Mitsubishi,
 * que salia como `MIlT5SUBlISHI` —la palabra se reconoce, pero con caracteres
 * insertados donde el modelo dudaba entre dos letras.
 *
 * Asi que el margen no se fija en pixeles: se calcula el que hace que la tinta
 * ocupe la misma proporcion con la que se entreno. No es ajustar el codigo a
 * una imagen, es presentarle al modelo lo que sabe leer.
 */
inline Bitmap RecortarRenglon(const Bitmap& imagen, const Renglon& r,
                              float proporcion = kProporcionTintaEntrenamiento) {
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

/** @brief Convierte un recorte en el tensor que espera el modelo. */
inline Tensor RenglonATensor(const Bitmap& recorte, int alto_objetivo, int multiplo_ancho,
                             bool invertir) {
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

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_RENGLONES_H_
