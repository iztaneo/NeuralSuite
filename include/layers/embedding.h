/**
 * @file embedding.h
 * @brief Capa de Embedding de Tokens y Posición para Transformers.
 */

#ifndef NEURAL_SUITE_EMBEDDING_H
#define NEURAL_SUITE_EMBEDDING_H

#include "../layer.h"

namespace ns {

class Embedding : public Layer {
public:
    int num_embeddings;
    int embedding_dim;
    Tensor weight;  // [num_embeddings, embedding_dim]
    Tensor dweight;
    std::vector<int> last_tokens;

    Embedding(int num_emb, int emb_dim)
        : num_embeddings(num_emb), embedding_dim(emb_dim),
          weight({num_emb, emb_dim}), dweight({num_emb, emb_dim}) {
        weight.random_normal(0.0f, 0.02f);
    }

    Tensor forward(const Tensor& input) override {
        int B = input.shape[0];
        int T = input.shape[1];
        last_tokens.resize(B * T);

        Tensor output({B, T, embedding_dim});

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                int tok = static_cast<int>(input.data[b * T + t]);
                last_tokens[b * T + t] = tok;

                for (int d = 0; d < embedding_dim; ++d) {
                    size_t out_idx = (b * T + t) * embedding_dim + d;
                    output.data[out_idx] = weight.data[tok * embedding_dim + d];
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        dweight.zeros();
        int B = dout.shape[0];
        int T = dout.shape[1];

        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                int tok = last_tokens[b * T + t];
                for (int d = 0; d < embedding_dim; ++d) {
                    size_t dout_idx = (b * T + t) * embedding_dim + d;
                    dweight.data[tok * embedding_dim + d] += dout.data[dout_idx];
                }
            }
        }
        return Tensor(); // No dx for discrete integer tokens
    }

    std::vector<Tensor*> get_parameters() override { return {&weight}; }
    std::vector<Tensor*> get_gradients() override { return {&dweight}; }
};

} // namespace ns

#endif // NEURAL_SUITE_EMBEDDING_H
