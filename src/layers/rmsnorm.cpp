// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/rmsnorm.h.

#include "layers/rmsnorm.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel.h"

namespace neuralsuite {

namespace {

// Cuantas filas hay y de que largo, dado que se normaliza el ultimo eje.
void Reparto(const Tensor& t, int esperado, int* filas, int* ancho) {
  const std::vector<int>& forma = t.Shape();
  if (forma.empty()) {
    throw std::invalid_argument("RMSNorm: la entrada no tiene ejes.");
  }
  *ancho = forma.back();
  if (*ancho != esperado) {
    throw std::invalid_argument(
        "RMSNorm: el ultimo eje mide " + std::to_string(*ancho) +
        " y la capa se creo para " + std::to_string(esperado) + ".");
  }
  *filas = static_cast<int>(t.TotalSize()) / *ancho;
}

}  // namespace

Tensor RMSNormLayer::Forward(const Tensor& input) {
  last_input_ = input;
  int filas = 0, ancho = 0;
  Reparto(input, normalized_shape_, &filas, &ancho);

  Tensor output(input.Shape());
  rms_cache_.Resize({filas});

  parallel::ParallelFor(filas, /*min_per_thread=*/16, [&](int desde, int hasta) {
    for (int f = desde; f < hasta; ++f) {
      const size_t base = static_cast<size_t>(f) * ancho;

      // La suma va en double: con ancho grande, acumular cuadrados en float
      // pierde precision justo donde el resultado alimenta una raiz.
      double suma = 0.0;
      for (int i = 0; i < ancho; ++i) {
        const double x = input[base + i];
        suma += x * x;
      }
      const float inv = static_cast<float>(1.0 / std::sqrt(suma / ancho + eps_));
      rms_cache_[f] = inv;

      for (int i = 0; i < ancho; ++i) {
        output[base + i] = input[base + i] * inv * gamma_.Value()[i];
      }
    }
  });
  return output;
}

Tensor RMSNormLayer::Backward(const Tensor& dout) {
  int filas = 0, ancho = 0;
  Reparto(last_input_, normalized_shape_, &filas, &ancho);

  Tensor dx(last_input_.Shape());
  Tensor& dgamma = gamma_.Grad();

  // Dos pasadas, y cada una reparte por un eje distinto. No es un capricho: la
  // cabecera de `ParallelFor` avisa de que el reparto es dinamico y de que no
  // hay correspondencia fija entre hilo y rango, asi que acumular en un vector
  // por hilo no vale. La salida de cada pasada es disjunta, no hay reduccion
  // entre hilos, y el resultado sale bit a bit igual con cualquier numero de
  // hilos.

  // 1. dx: cada fila escribe su propio tramo.
  parallel::ParallelFor(filas, /*min_per_thread=*/16, [&](int desde, int hasta) {
    for (int f = desde; f < hasta; ++f) {
      const size_t base = static_cast<size_t>(f) * ancho;
      const float inv = rms_cache_[f];

      // suma(dout * gamma * x) sobre la fila. Es el termino que hace ortogonal
      // el gradiente respecto a x, y el que se olvida al derivar a mano: sin el
      // la red sigue entrenando, algo peor, y ninguna prueba que no compruebe
      // el gradiente lo nota.
      double proyeccion = 0.0;
      for (int i = 0; i < ancho; ++i) {
        proyeccion += static_cast<double>(dout[base + i]) * gamma_.Value()[i] *
                      last_input_[base + i];
      }
      const float escala = static_cast<float>(proyeccion) * inv * inv * inv /
                           static_cast<float>(ancho);

      for (int i = 0; i < ancho; ++i) {
        dx[base + i] = dout[base + i] * gamma_.Value()[i] * inv -
                       last_input_[base + i] * escala;
      }
    }
  });

  // 2. dgamma: se reparte por COLUMNAS. Cada gamma[i] suma sobre todas las
  //    filas, asi que repartir por filas obligaria a reducir entre hilos;
  //    repartiendo por columnas cada hilo escribe posiciones distintas y suma
  //    las filas en orden fijo.
  parallel::ParallelFor(ancho, /*min_per_thread=*/64, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      double acc = 0.0;
      for (int f = 0; f < filas; ++f) {
        const size_t k = static_cast<size_t>(f) * ancho + i;
        acc += static_cast<double>(dout[k]) * last_input_[k] * rms_cache_[f];
      }
      dgamma[i] = static_cast<float>(acc);   // asigna, como hacen las demas capas
    }
  });

  return dx;
}

}  // namespace neuralsuite
