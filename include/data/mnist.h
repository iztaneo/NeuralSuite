// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file mnist.h
 * @brief Lector del formato IDX, en el que se distribuye MNIST.
 */

#ifndef NEURAL_SUITE_INCLUDE_DATA_MNIST_H_
#define NEURAL_SUITE_INCLUDE_DATA_MNIST_H_

#include <cstdint>
#include <string>
#include <vector>

#include "../tensor.h"

namespace neuralsuite {
namespace data {

/**
 * @struct ConjuntoMnist
 * @brief Imagenes y etiquetas ya en memoria.
 *
 * `imagenes` es `[N, 1, 28, 28]` con valores en `[0, 1]`, que es la forma que
 * esperan `Conv2D` y las capas de vision. `etiquetas` es `[N]` con el digito.
 */
struct ConjuntoMnist {
  Tensor imagenes;
  Tensor etiquetas;
  int n = 0;

  [[nodiscard]] bool Vacio() const { return n == 0; }
};

/**
 * @brief Lee un par de archivos IDX (imagenes y etiquetas) de MNIST.
 *
 * El formato IDX es deliberadamente simple, y su unica trampa es que **los
 * enteros van en big-endian**: el archivo se escribio en una epoca en que eso
 * era comun, y leerlos como little-endian —que es lo nativo aqui— da numeros
 * absurdos como 50 331 648 imagenes en vez de 60 000. Por eso la cabecera se
 * lee byte a byte y no con un `memcpy`.
 *
 *     imagenes:   magic 0x00000803, n, filas, columnas, luego n*filas*cols bytes
 *     etiquetas:  magic 0x00000801, n, luego n bytes
 *
 * Se comprueban las dos cosas que pueden salir mal en silencio: que el numero
 * magico sea el que corresponde —abrir el archivo de etiquetas como si fuera el
 * de imagenes es el error facil, porque los nombres se parecen— y que las dos
 * cuentas coincidan. Un desajuste ahi entrenaria con etiquetas corridas y el
 * modelo simplemente no aprenderia, sin que nada fallara.
 *
 * @param normalizar si es cierto, los pixeles se dividen entre 255.
 * @return conjunto vacio y `error` explicado si algo no encaja.
 */
bool LeerMnist(const std::string& ruta_imagenes, const std::string& ruta_etiquetas,
               ConjuntoMnist* salida, std::string* error, bool normalizar = true);

/** @brief Lee un archivo IDX de imagenes suelto, sin etiquetas. */
bool LeerIdxImagenes(const std::string& ruta, Tensor* salida, int* n, std::string* error,
                     bool normalizar = true);

/** @brief Lee un archivo IDX de etiquetas suelto. */
bool LeerIdxEtiquetas(const std::string& ruta, Tensor* salida, int* n, std::string* error);

}  // namespace data
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DATA_MNIST_H_
