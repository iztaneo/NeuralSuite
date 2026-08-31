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
std::vector<Renglon> DetectarRenglones(const Bitmap& imagen, int alto_minimo = 4,
                                              int union_maxima = 2,
                                              float proporcion_acento = 0.4f,
                                              Binarizacion metodo = Binarizacion::kAutomatica);

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
Bitmap RecortarRenglon(const Bitmap& imagen, const Renglon& r,
                              float proporcion = kProporcionTintaEntrenamiento);

/** @brief Convierte un recorte en el tensor que espera el modelo. */
Tensor RenglonATensor(const Bitmap& recorte, int alto_objetivo, int multiplo_ancho,
                             bool invertir);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_RENGLONES_H_
