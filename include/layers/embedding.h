// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file embedding.h
 * @brief Token & Position Embedding Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class Embedding
 * @brief Maps discrete integer token indices to continuous embedding vectors.
 */
class Embedding : public Layer {
 public:
  Embedding(int num_emb, int emb_dim)
      : num_embeddings_(num_emb),
        embedding_dim_(emb_dim),
        weight_({num_emb, emb_dim}),
        dweight_({num_emb, emb_dim}) {
    weight_.RandomNormal(0.0f, 0.02f);
  }

  Tensor Forward(const Tensor& input) override {
    int batch_size = input.Shape()[0];
    int seq_len = input.Shape()[1];
    last_tokens_.resize(batch_size * seq_len);

    Tensor output({batch_size, seq_len, embedding_dim_});

    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        int tok = static_cast<int>(input[b * seq_len + t]);
        last_tokens_[b * seq_len + t] = tok;

        for (int d = 0; d < embedding_dim_; ++d) {
          size_t out_idx = (b * seq_len + t) * embedding_dim_ + d;
          output[out_idx] = weight_[tok * embedding_dim_ + d];
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    dweight_.Zeros();
    int batch_size = dout.Shape()[0];
    int seq_len = dout.Shape()[1];

    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        int tok = last_tokens_[b * seq_len + t];
        for (int d = 0; d < embedding_dim_; ++d) {
          size_t dout_idx = (b * seq_len + t) * embedding_dim_ + d;
          dweight_[tok * embedding_dim_ + d] += dout[dout_idx];
        }
      }
    }
    return Tensor();
  }

  std::vector<Tensor*> GetParameters() override { return {&weight_}; }
  std::vector<Tensor*> GetGradients() override { return {&dweight_}; }

  [[nodiscard]] Tensor& Weight() { return weight_; }

 private:
  int num_embeddings_;
  int embedding_dim_;

  Tensor weight_;
  Tensor dweight_;
  std::vector<int> last_tokens_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_EMBEDDING_H_
