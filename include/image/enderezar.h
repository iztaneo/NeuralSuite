// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file enderezar.h
 * @brief Estima y corrige la inclinacion de una pagina escaneada.
 *
 * La separacion en renglones se apoya en la proyeccion horizontal: contar la
 * tinta de cada fila y buscar las filas vacias que separan un renglon del
 * siguiente. Eso supone que los renglones son horizontales, y deja de valer en
 * cuanto no lo son: con la pagina inclinada, una fila de pixeles atraviesa
 * varios renglones y las filas vacias desaparecen.
 *
 * No es un caso raro. Medido sobre la pagina de la Iliada de este repositorio:
 *
 *   original      18 renglones     Tesseract 0.1% de error
 *   torcida 2º     4 renglones     Tesseract 0.8%
 *   torcida 5º     3 renglones     Tesseract 5.2%
 *
 * Dos grados es lo que sale de poner una hoja a mano en un escaner, y bastan
 * para tirar la segmentacion. Tesseract los absorbe porque tiene esta etapa.
 *
 * El metodo: no se busca «el angulo del texto» sino **el angulo que hace la
 * proyeccion mas nitida**. Cuando los renglones estan horizontales, la
 * proyeccion alterna filas llenas con filas vacias y su varianza es maxima; al
 * inclinarse, los picos se aplanan. Se prueban angulos y se toma el mejor.
 *
 * Optimizar la nitidez de la proyeccion, y no una nocion abstracta de
 * alineacion, tiene una ventaja concreta: es exactamente la magnitud de la que
 * depende el cortador de renglones. Si algun dia el cortador cambia de
 * criterio, esta funcion seguira midiendo lo que a el le importa.
 *
 * **La precision depende del ancho de la imagen**, y conviene saberlo. Un
 * grado decimo inclina la ultima columna respecto de la primera en
 * `ancho * tan(0.1º)` pixeles: 0.35 en una imagen de 200 px de ancho, 1.4 en
 * una de 800. Por debajo de un pixel no hay nada que la proyeccion pueda
 * distinguir. Medido sobre una pagina sintetica perfectamente recta, el
 * estimador devuelve -0.30º con 200 px de ancho, -0.10º con 400 y 0.00º a
 * partir de 800. En un escaneo real, que ronda los mil pixeles, la resolucion
 * de una decima esta holgada; en una miniatura, no.
 *
 * Se descarto la transformada de Hough sobre bordes: es mas general, pero pesa
 * mas y necesita deteccion de bordes con sus propios parametros. Y ajustar una
 * recta a los centroides de componentes conexas exige etiquetarlas primero.
 * Para texto en renglones, la proyeccion no es peor y es mucho mas corta.
 */

#ifndef NEURAL_SUITE_INCLUDE_IMAGE_ENDEREZAR_H_
#define NEURAL_SUITE_INCLUDE_IMAGE_ENDEREZAR_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "../parallel.h"
#include "bitmap.h"
#include "renglones.h"

namespace neuralsuite {
namespace image {

/**
 * @brief Nitidez de la proyeccion horizontal para un angulo dado.
 *
 * Se acumula la tinta de cada fila **como si** la imagen estuviera girada,
 * sin girarla: para cada pixel se calcula en que fila caeria. Girar de verdad
 * en cada prueba costaria una interpolacion completa por angulo.
 *
 * La medida es la suma de los cuadrados de las diferencias entre filas
 * consecutivas. Aqui hay que ser honesto: al escribir esto se justifico la
 * eleccion diciendo que la varianza tambien subiria con la tinta concentrada en
 * una banda ancha —una ilustracion, un margen oscuro— y que la diferencia entre
 * vecinas solo responde a la alternancia lleno/vacio. **Se comprobo y no es
 * cierto**: sobre diez casos, con y sin un bloque macizo que ocupa un tercio de
 * la pagina, las dos medidas devuelven el mismo angulo hasta la decima.
 *
 * Asi que la eleccion no esta justificada por esa razon. Se conserva la
 * diferencia entre vecinas porque no necesita una pasada previa para calcular
 * la media, y porque es lo que se probo; si alguien encuentra un caso donde
 * discrepen, ese caso decide.
 */
inline double NitidezProyeccion(const std::vector<uint8_t>& tinta, int ancho, int alto,
                                double radianes) {
  const double sen = std::sin(radianes);
  std::vector<double> filas(static_cast<size_t>(alto), 0.0);
  const double centro_x = ancho * 0.5, centro_y = alto * 0.5;

  for (int y = 0; y < alto; ++y) {
    const uint8_t* linea = tinta.data() + static_cast<size_t>(y) * ancho;
    for (int x = 0; x < ancho; ++x) {
      if (!linea[x]) continue;
      // Solo interesa la coordenada vertical tras el giro.
      const double fy = centro_y + (y - centro_y) - (x - centro_x) * sen;
      const int destino = static_cast<int>(fy + 0.5);
      if (destino >= 0 && destino < alto) filas[destino] += 1.0;
    }
  }

  double suma = 0.0;
  for (int y = 1; y < alto; ++y) {
    const double d = filas[y] - filas[y - 1];
    suma += d * d;
  }
  return suma;
}

/**
 * @brief Angulo de inclinacion en grados, positivo en sentido antihorario.
 *
 * La busqueda es en dos pasadas: una gruesa por todo el rango y otra fina
 * alrededor del mejor. Probar directamente en pasos de una decima costaria diez
 * veces mas evaluaciones para el mismo resultado.
 */
inline double EstimarInclinacion(const Bitmap& imagen, double rango_grados = 8.0,
                                 double paso_grueso = 0.5, double paso_fino = 0.1) {
  if (imagen.Empty()) return 0.0;

  std::vector<float> gris;
  ToGrayscale(imagen, &gris);
  const int ancho = imagen.width, alto = imagen.height;
  const int umbral = UmbralOtsu(gris);

  // Se binariza una vez y se reutiliza en todas las pruebas: lo que cambia
  // entre angulos es donde cae cada pixel, no cual es tinta.
  std::vector<uint8_t> tinta(static_cast<size_t>(ancho) * alto, 0);
  for (size_t i = 0; i < tinta.size(); ++i) {
    tinta[i] = static_cast<int>(gris[i] * 255.0f + 0.5f) <= umbral ? 1 : 0;
  }

  const auto buscar = [&](double desde, double hasta, double paso) {
    const int n = static_cast<int>(std::lround((hasta - desde) / paso)) + 1;
    std::vector<double> puntuacion(static_cast<size_t>(std::max(1, n)), 0.0);
    parallel::ParallelFor(n, /*min_per_thread=*/1, [&](int a, int b) {
      for (int i = a; i < b; ++i) {
        puntuacion[i] = NitidezProyeccion(tinta, ancho, alto,
                                          (desde + i * paso) * kPi / 180.0);
      }
    });
    int mejor = 0;
    for (int i = 1; i < n; ++i) {
      if (puntuacion[i] > puntuacion[mejor]) mejor = i;
    }
    return desde + mejor * paso;
  };

  const double grueso = buscar(-rango_grados, rango_grados, paso_grueso);
  return buscar(grueso - paso_grueso, grueso + paso_grueso, paso_fino);
}

/**
 * @brief Gira la imagen por interpolacion bilineal.
 *
 * Se recorre el destino y se busca de donde viene cada pixel, no al reves:
 * recorrer el origen dejaria huecos sin escribir. Lo que cae fuera se rellena
 * con el nivel del fondo, estimado como el mas frecuente de los bordes —usar
 * blanco daria un marco luminoso en un escaneo gris, y ese marco entraria en el
 * histograma de Otsu falseando el umbral.
 */
inline Bitmap Girar(const Bitmap& imagen, double grados) {
  Bitmap salida;
  salida.width = imagen.width;
  salida.height = imagen.height;
  salida.channels = imagen.channels;
  const int C = imagen.channels;

  // Nivel del fondo: la mediana del borde, por canal.
  std::vector<uint8_t> fondo(static_cast<size_t>(C), 255);
  for (int c = 0; c < C; ++c) {
    std::vector<uint8_t> muestras;
    for (int x = 0; x < imagen.width; ++x) {
      muestras.push_back(imagen.pixels[static_cast<size_t>(x) * C + c]);
      muestras.push_back(
          imagen.pixels[(static_cast<size_t>(imagen.height - 1) * imagen.width + x) * C + c]);
    }
    if (!muestras.empty()) {
      std::nth_element(muestras.begin(), muestras.begin() + muestras.size() / 2, muestras.end());
      fondo[c] = muestras[muestras.size() / 2];
    }
  }

  salida.pixels.assign(static_cast<size_t>(salida.width) * salida.height * C, 0);
  const double radianes = grados * kPi / 180.0;
  const double sen = std::sin(radianes), cos_ = std::cos(radianes);
  const double cx = imagen.width * 0.5, cy = imagen.height * 0.5;

  parallel::ParallelFor(salida.height, /*min_per_thread=*/8, [&](int desde, int hasta) {
    for (int y = desde; y < hasta; ++y) {
      for (int x = 0; x < salida.width; ++x) {
        // Giro inverso: de donde viene este pixel del destino.
        const double dx = x - cx, dy = y - cy;
        const double ox = cx + dx * cos_ + dy * sen;
        const double oy = cy - dx * sen + dy * cos_;

        uint8_t* destino = salida.pixels.data() + (static_cast<size_t>(y) * salida.width + x) * C;
        if (ox < 0 || oy < 0 || ox > imagen.width - 1 || oy > imagen.height - 1) {
          for (int c = 0; c < C; ++c) destino[c] = fondo[c];
          continue;
        }
        const int x0 = static_cast<int>(ox), y0 = static_cast<int>(oy);
        const int x1 = std::min(x0 + 1, imagen.width - 1);
        const int y1 = std::min(y0 + 1, imagen.height - 1);
        const double wx = ox - x0, wy = oy - y0;

        for (int c = 0; c < C; ++c) {
          const double a = imagen.pixels[(static_cast<size_t>(y0) * imagen.width + x0) * C + c];
          const double b = imagen.pixels[(static_cast<size_t>(y0) * imagen.width + x1) * C + c];
          const double d = imagen.pixels[(static_cast<size_t>(y1) * imagen.width + x0) * C + c];
          const double e = imagen.pixels[(static_cast<size_t>(y1) * imagen.width + x1) * C + c];
          const double v = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + d * (1 - wx) * wy +
                           e * wx * wy;
          destino[c] = static_cast<uint8_t>(std::lround(std::max(0.0, std::min(255.0, v))));
        }
      }
    }
  });
  return salida;
}

/**
 * @brief Estima la inclinacion y devuelve la imagen enderezada.
 *
 * Por debajo de `minimo_grados` no se gira: una rotacion cuesta una
 * interpolacion de toda la imagen y emborrona un poco los trazos, asi que por
 * media decima de grado sale mas caro el remedio.
 */
inline Bitmap Enderezar(const Bitmap& imagen, double* grados_aplicados = nullptr,
                        double minimo_grados = 0.15) {
  const double grados = EstimarInclinacion(imagen);
  if (grados_aplicados) *grados_aplicados = grados;
  if (std::abs(grados) < minimo_grados) return imagen;
  return Girar(imagen, -grados);
}

}  // namespace image
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_IMAGE_ENDEREZAR_H_
