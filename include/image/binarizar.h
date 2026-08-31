// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file binarizar.h
 * @brief Separar tinta de fondo, con un umbral global o uno por vecindad.
 *
 * El metodo de Otsu calcula **un** umbral para toda la imagen, buscando el
 * corte que deja las dos poblaciones de pixeles lo mas separadas posible. Va
 * bien en un escaneo parejo y falla en cuanto la iluminacion no lo es: con un
 * degradado lateral —una foto de una pagina con luz de un lado— medio folio
 * cae entero por debajo del umbral y se cuenta como tinta. Medido sobre la
 * pagina de la Iliada con un degradado del 55%, la separacion en renglones
 * pasaba de 18 a 1.
 *
 * Sauvola calcula un umbral por pixel a partir de la media `m` y la desviacion
 * `s` de su vecindad:
 *
 *     T(x,y) = m · [1 + k · (s/R − 1)]
 *
 * En una ventana de puro fondo la desviacion es baja, `T` queda por debajo de
 * la media y no se marca nada. En una ventana con texto la desviacion es alta,
 * `T` sube hasta la media y la mitad oscura pasa a ser tinta. Se adapta al
 * brillo local sin que nadie le diga cual es.
 *
 * Calcularlo pixel a pixel recorriendo su ventana seria O(ventana²) por pixel.
 * Con imagenes integrales —tablas de sumas acumuladas de los valores y de sus
 * cuadrados— la media y la varianza de cualquier rectangulo salen con cuatro
 * lecturas, y el coste pasa a ser constante.
 *
 * **El tamano de la ventana es el parametro delicado**, y conviene decirlo. Deberia
 * ser del orden del alto del texto, pero el alto del texto no se conoce hasta
 * haber segmentado: pez que se muerde la cola. Se toma una fraccion de la
 * dimension menor de la imagen, con limites, y se midio la sensibilidad antes
 * de fijarla (ver la tabla en docs/ROADMAP.md). No es un numero elegido a ojo,
 * pero tampoco es una deduccion: es el que resiste mejor el barrido.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_BINARIZAR_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_BINARIZAR_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "../parallel.h"
#include "bitmap.h"

namespace neuralsuite {
namespace image {

/** @brief Umbral global por el metodo de Otsu. Devuelve un corte inclusive. */
int UmbralOtsu(const std::vector<float>& gris);

/** @brief Marca la tinta con un umbral global. 1 es tinta, 0 es fondo. */
std::vector<uint8_t> BinarizarGlobal(const std::vector<float>& gris);

/**
 * @brief Ventana que usa Sauvola, deducida del tamano de la imagen.
 *
 * Impar siempre, para que tenga centro. Los limites evitan los dos extremos
 * inutiles: una ventana diminuta responde al grano del papel y una enorme se
 * comporta como un umbral global, que es justo lo que se quiere evitar.
 */
int VentanaSauvola(int ancho, int alto);

/**
 * @brief Marca la tinta con un umbral por vecindad (Sauvola).
 *
 * `k` controla cuanto se aparta el umbral de la media local: mas alto marca
 * menos tinta. 0.2 es el valor del articulo original y el que usan las
 * implementaciones al uso; `R` es el rango dinamico de la desviacion, 128 para
 * imagenes de ocho bits.
 *
 * Una honestidad sobre lo que aqui esta comprobado y lo que no. **El termino de
 * la desviacion local no se distingue en ninguna de las imagenes que se
 * prueban**: mutar la imagen integral de cuadrados para que acumule los valores
 * sin elevar —lo que anula ese termino y deja el umbral en 0.8 veces la media
 * local— no cambia ni un renglon en la pagina de la Iliada, ni con sombra, ni
 * con ruido, ni en el abecedario manuscrito.
 *
 * No se simplifica a la media local por eso. Cuatro imagenes no bastan para
 * descartar un termino que la literatura justifica para documentos con trazos
 * de contraste variable, y quitarlo seria optimizar contra la muestra que se
 * tiene. Pero conviene saber que ese termino no esta verificado aqui: si algun
 * dia aparece un caso donde importe, ese caso es el que hay que anadir.
 */
std::vector<uint8_t> BinarizarSauvola(const std::vector<float>& gris, int ancho, int alto,
                                             int ventana = 0, double k = 0.2, double R = 128.0);

/** @brief Como se separa la tinta del fondo. */
enum class Binarizacion { kGlobal, kAdaptativa, kAutomatica };

/**
 * @brief Nitidez de la proyeccion horizontal de una imagen binarizada.
 *
 * Cuenta la tinta de cada fila y suma los cuadrados de las diferencias entre
 * filas consecutivas: crece cuando las filas llenas alternan con las vacias,
 * que es exactamente lo que la separacion en renglones necesita.
 *
 * Se divide por la tinta total, y eso no es un detalle: sin normalizar, marcar
 * mas pixeles siempre daria mas nitidez y ganaria el metodo mas permisivo
 * aunque emborronara los renglones. Comprobado quitando la division: la pagina
 * de la Iliada con ruido de escaneo baja de 18 renglones a 16, porque el
 * umbral adaptativo marca de mas sobre el grano y aun asi gana.
 *
 * Ese efecto **no lo reproduce ninguna prueba sintetica** de las que hay. Se
 * intento con bandas macizas y con trazos finos, y en las dos la mutacion
 * pasaba; seguir ajustando la imagen de prueba hasta que fallara habria sido
 * escribir una prueba para cazar una mutacion concreta, no para encontrar
 * defectos. Queda anotado como reproducirlo a mano:
 *
 *     tools/image/... genera 05_ruido.png a partir de iliada_libro1.png
 *     DetectarRenglones sobre ella debe dar 18; sin normalizar da 16.
 */
double NitidezDeFilas(const std::vector<uint8_t>& marca, int ancho, int alto);

/**
 * @brief Marca la tinta con el metodo pedido.
 *
 * `kAutomatica` prueba los dos y se queda con el que deja la proyeccion mas
 * nitida. Hace falta porque **ninguno de los dos es mejor que el otro**, y eso
 * se midio antes de decidirlo. Sobre la pagina de la Iliada y sus variantes:
 *
 *   original         Otsu 18 renglones   Sauvola 18
 *   con sombra       Otsu  1             Sauvola 18
 *   con ruido        Otsu 18             Sauvola 16
 *   bajo contraste   Otsu  1             Sauvola  0
 *
 * Otsu se hunde con la iluminacion desigual, que es justo para lo que existe
 * Sauvola; y Sauvola se hunde con el ruido y con el contraste bajo, porque su
 * umbral se apoya en la desviacion local y ahi es pequena. Fijar uno de los dos
 * seria equivocarse en algun caso.
 *
 * Elegir por la nitidez de la proyeccion no es una heuristica ajustada a estas
 * imagenes: es el mismo principio que usa el enderezado, optimizar la magnitud
 * de la que depende el cortador. Comprobado en los cinco casos, el metodo con
 * mas nitidez es el que da el numero correcto de renglones.
 */
std::vector<uint8_t> Binarizar(const std::vector<float>& gris, int ancho, int alto,
                                      Binarizacion metodo);

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_BINARIZAR_H_
