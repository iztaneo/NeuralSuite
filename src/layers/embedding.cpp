// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/embedding.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/embedding.h"

namespace neuralsuite {

Tensor Embedding::Forward(const Tensor& input) {
    int batch_size = input.Shape()[0];
    int seq_len = input.Shape()[1];
    last_tokens_.resize(batch_size * seq_len);

    Tensor output({batch_size, seq_len, embedding_dim_});

    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        int tok = static_cast<int>(input[b * seq_len + t]);
        // Un token fuera de rango indexaría weight_ fuera de su memoria.
        if (tok < 0 || tok >= num_embeddings_) {
          throw std::out_of_range(
              "Embedding: token " + std::to_string(tok) + " fuera del rango [0, " +
              std::to_string(num_embeddings_) + ").");
        }
        last_tokens_[b * seq_len + t] = tok;

        for (int d = 0; d < embedding_dim_; ++d) {
          size_t out_idx = (b * seq_len + t) * embedding_dim_ + d;
          output[out_idx] = weight_.Value()[tok * embedding_dim_ + d];
        }
      }
    }
    return output;
  }

Tensor Embedding::Backward(const Tensor& dout) {
    Tensor& dweight = weight_.Grad();
    dweight.Zeros();
    int batch_size = dout.Shape()[0];
    int seq_len = dout.Shape()[1];
    size_t num_cached = last_tokens_.size();
    if (num_cached == 0) return Tensor();

    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        size_t cache_idx = (b * seq_len + t) % num_cached;
        int tok = last_tokens_[cache_idx];
        for (int d = 0; d < embedding_dim_; ++d) {
          size_t dout_idx = (b * seq_len + t) * embedding_dim_ + d;
          dweight[tok * embedding_dim_ + d] += dout[dout_idx];
        }
      }
    }
    return Tensor();
  }

}  // namespace neuralsuite
