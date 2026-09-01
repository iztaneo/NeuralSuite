// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file embedding_autograd.h
 * @brief El mismo embedding, pero derivado por el grafo en vez de a mano.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_AUTOGRAD_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_AUTOGRAD_H_

#include <vector>
#include "../autograd.h"
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class EmbeddingAutograd
 * @brief Filas de una tabla por indice, con el gradiente obtenido del grafo.
 *
 * Intercambiable con `Embedding`: mismo constructor, misma inicializacion y el
 * mismo parametro registrado, de modo que con la misma semilla las dos parten
 * de pesos identicos.
 *
 * Toda la capa es una llamada a `Gather`. El backward —que en `Embedding`
 * ocupa un bucle anidado con una acumulacion por token— aqui no existe: lo
 * pone la primitiva.
 *
 * Segunda pareja del esquema que abrio `LinearAutograd`, y por la misma razon:
 * dos caminos independientes al mismo numero, con una prueba que los enfrenta.
 * `Embedding` sigue siendo la de entrenar; esta es la que dice si acierta.
 */
class EmbeddingAutograd : public Layer {
 public:
  EmbeddingAutograd(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim),
        weight_({num_embeddings, embedding_dim}) {
    Register(&weight_, "weight");
    weight_.Value().RandomNormal(0.0f, 0.02f);
  }

  Tensor Forward(const Tensor& input) override;

  /**
   * @brief Propaga hacia la tabla. Devuelve un tensor vacio, como `Embedding`.
   *
   * La entrada son indices enteros, no un valor continuo: no hay gradiente que
   * devolver hacia atras, solo el de la tabla.
   */
  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] Tensor& Weight() { return weight_.Value(); }
  [[nodiscard]] Parameter& WeightParam() { return weight_; }

 private:
  int num_embeddings_;
  int embedding_dim_;

  Parameter weight_;

  autograd::VarPtr tabla_;
  autograd::VarPtr salida_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_AUTOGRAD_H_
