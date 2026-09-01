// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/embedding_autograd.h.

#include "layers/embedding_autograd.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuralsuite {

Tensor EmbeddingAutograd::Forward(const Tensor& input) {
  if (input.Shape().size() != 2) {
    throw std::invalid_argument(
        "EmbeddingAutograd: la entrada debe ser [lote, longitud].");
  }

  if (weight_.Value().Shape() != std::vector<int>({num_embeddings_, embedding_dim_})) {
    throw std::invalid_argument(
        "EmbeddingAutograd: la tabla no es [vocabulario, dimension].");
  }

  // La tabla cambia tras cada paso del optimizador, asi que la hoja del grafo
  // se crea de nuevo en cada pasada a partir del valor vigente.
  tabla_ = autograd::Variable::Create(weight_.Value(), /*requires_grad=*/true);

  // Aqui esta toda la capa. `Gather` valida que cada indice caiga dentro de la
  // tabla, igual que hace `Embedding`.
  salida_ = autograd::Gather(tabla_, input);
  return salida_->Value();
}

Tensor EmbeddingAutograd::Backward(const Tensor& dout) {
  // `Backward` exige una raiz escalar y siembra el gradiente el mismo; se cierra
  // el grafo con una perdida cuya derivada respecto a la salida es ese `dout`.
  autograd::Backward(autograd::Sum(
      autograd::Mul(salida_, autograd::Variable::Create(dout))));

  // `Embedding` pone a cero y acumula, o sea que asigna; se copia igual para
  // que las dos capas sean intercambiables ante el optimizador.
  std::memcpy(weight_.Grad().Data(), tabla_->Grad().Data(),
              weight_.Grad().TotalSize() * sizeof(float));

  return Tensor();
}

}  // namespace neuralsuite
