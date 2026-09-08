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
  // Se siembra el gradiente directamente, sin materializar un tensor del tamano
  // de la salida solo para propagarlo.
  //
  // Ojo con el caso del GPT: el embedding de posicion se calcula con forma
  // [1, T] y su gradiente llega [B, T, D]. Antes eso lo resolvia el broadcasting
  // de `Mul`; ahora hay que reducirlo aqui, sumando sobre el lote, que es la
  // misma operacion escrita donde se ve.
  Tensor semilla = dout;
  if (semilla.Shape() != salida_->Shape()) {
    const size_t n = salida_->Value().TotalSize();
    if (semilla.TotalSize() % n != 0) {
      throw std::invalid_argument(
          "EmbeddingAutograd: el gradiente no encaja con la salida.");
    }
    const size_t repeticiones = semilla.TotalSize() / n;
    Tensor sumado(salida_->Shape());
    sumado.Zeros();
    for (size_t r = 0; r < repeticiones; ++r) {
      for (size_t i = 0; i < n; ++i) sumado[i] += dout[r * n + i];
    }
    semilla = std::move(sumado);
  }
  autograd::Backward(salida_, semilla);

  // `Embedding` pone a cero y acumula, o sea que asigna; se copia igual para
  // que las dos capas sean intercambiables ante el optimizador.
  std::memcpy(weight_.Grad().Data(), tabla_->Grad().Data(),
              weight_.Grad().TotalSize() * sizeof(float));

  return Tensor();
}

}  // namespace neuralsuite
