// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file embedding.h
 * @brief Token & Position Embedding Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_

#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class Embedding
 * @brief Maps discrete integer token indices to continuous embedding vectors.
 */
class Embedding : public Layer {
 public:
  Embedding(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim),
        weight_({num_embeddings, embedding_dim}) {
    Register(&weight_, "weight");
    weight_.Value().RandomNormal(0.0f, 0.02f);
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;


  [[nodiscard]] Tensor& Weight() { return weight_.Value(); }
  [[nodiscard]] Parameter& WeightParam() { return weight_; }

 private:
  int num_embeddings_;
  int embedding_dim_;

  Parameter weight_;
  std::vector<int> last_tokens_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_
