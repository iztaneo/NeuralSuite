// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de diffusion/time_embedding.h.

#include "diffusion/time_embedding.h"

#include <cmath>

namespace neuralsuite {
namespace diffusion {

void TimeEmbedding(const Tensor& t, int dim, Tensor* salida, float base) {
  if (dim <= 0 || dim % 2 != 0) {
    throw std::invalid_argument(
        "TimeEmbedding: la dimension debe ser par y positiva; vale " +
        std::to_string(dim) + ".");
  }
  if (t.Shape().empty()) {
    throw std::invalid_argument("TimeEmbedding: el tensor de pasos no tiene ejes.");
  }

  const int n = static_cast<int>(t.TotalSize());
  const int mitad = dim / 2;
  salida->Resize({n, dim});

  for (int i = 0; i < n; ++i) {
    const double paso = static_cast<double>(t[i]);
    const size_t fila = static_cast<size_t>(i) * dim;
    for (int k = 0; k < mitad; ++k) {
      // La frecuencia decae exponencialmente con el indice: el primer canal gira
      // una vez por paso y el ultimo tarda ~10000 pasos en dar la vuelta. Esa
      // escala es la que hace que pasos cercanos den vectores parecidos.
      const double frec = std::exp(-std::log(static_cast<double>(base)) *
                                   static_cast<double>(k) / mitad);
      const double ang = paso * frec;
      // Primero TODOS los senos, luego TODOS los cosenos. Intercalarlos da un
      // embedding igual de valido y que no coincide con ninguna referencia.
      (*salida)[fila + k] = static_cast<float>(std::sin(ang));
      (*salida)[fila + mitad + k] = static_cast<float>(std::cos(ang));
    }
  }
}

}  // namespace diffusion
}  // namespace neuralsuite
