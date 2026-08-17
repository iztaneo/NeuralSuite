/**
 * @file attention.h
 * @brief Capa de Atención Causal Multi-Cabeza (Multi-Head Self-Attention) del paper "Attention Is All You Need".
 */

#ifndef NEURAL_SUITE_ATTENTION_H
#define NEURAL_SUITE_ATTENTION_H

#include "../layer.h"

namespace ns {

/**
 * @class MultiHeadAttention
 * @brief Mecanismo de Atención Causal Multi-Cabeza para Transformers Decoder-Only (GPT).
 * @details Calcula Q, K, V = X * W_qkv, luego Attn = Softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V.
 */
class MultiHeadAttention : public Layer {
public:
    int n_embd;   ///< Dimensión total del modelo d_model
    int n_head;   ///< Número de cabezas de atención h
    int head_dim; ///< Dimensión de cada cabeza d_k = d_model / h

    Tensor c_attn_weight;  ///< Pesos combinados Q, K, V [3 * n_embd, n_embd]
    Tensor c_attn_bias;    ///< Sesgos combinados Q, K, V [3 * n_embd]
    Tensor c_proj_weight;  ///< Pesos de proyección final de salida [n_embd, n_embd]
    Tensor c_proj_bias;    ///< Sesgos de proyección final [n_embd]

    Tensor dc_attn_weight;
    Tensor dc_attn_bias;
    Tensor dc_proj_weight;
    Tensor dc_proj_bias;

    Tensor last_input;

    MultiHeadAttention(int embd, int heads)
        : n_embd(embd), n_head(heads), head_dim(embd / heads),
          c_attn_weight({3 * embd, embd}), c_attn_bias({3 * embd}),
          c_proj_weight({embd, embd}), c_proj_bias({embd}),
          dc_attn_weight({3 * embd, embd}), dc_attn_bias({3 * embd}),
          dc_proj_weight({embd, embd}), dc_proj_bias({embd}) {
        c_attn_weight.xavier_init(embd, 3 * embd);
        c_proj_weight.xavier_init(embd, embd);
        c_attn_bias.zeros();
        c_proj_bias.zeros();
    }

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int B = input.shape[0];
        int T = input.shape[1];

        // 1. Proyección conjunta Q, K, V
        Tensor qkv({B * T, 3 * n_embd});
        Tensor input_2d({B * T, n_embd});
        std::memcpy(input_2d.data, input.data, input.total_size() * sizeof(float));

        Tensor W_T = transpose(c_attn_weight);
        matmul(input_2d, W_T, qkv);

        // 2. Cálculo de Atención Causal dividida por cabezas
        Tensor output({B, T, n_embd});
        output.zeros();

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head; ++h) {
                Tensor scores({T, T});
                scores.zeros();

                for (int i = 0; i < T; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            size_t q_idx = (b * T + i) * (3 * n_embd) + (h * head_dim + d);
                            size_t k_idx = (b * T + j) * (3 * n_embd) + n_embd + (h * head_dim + d);
                            dot += qkv.data[q_idx] * qkv.data[k_idx];
                        }
                        scores.data[i * T + j] = dot * scale;
                    }
                }

                // Causal Softmax
                Tensor attn({T, T});
                causal_softmax_forward(scores, attn, T);

                // Multiplicar por Valores V
                for (int i = 0; i < T; ++i) {
                    for (int d = 0; d < head_dim; ++d) {
                        float val = 0.0f;
                        for (int j = 0; j <= i; ++j) {
                            size_t v_idx = (b * T + j) * (3 * n_embd) + 2 * n_embd + (h * head_dim + d);
                            val += attn.data[i * T + j] * qkv.data[v_idx];
                        }
                        size_t out_idx = (b * T + i) * n_embd + (h * head_dim + d);
                        output.data[out_idx] = val;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        Tensor dx(last_input.shape);
        dx.zeros();
        dc_attn_weight.zeros(); dc_attn_bias.zeros();
        dc_proj_weight.zeros(); dc_proj_bias.zeros();
        return dx;
    }

    std::vector<Tensor*> get_parameters() override { return {&c_attn_weight, &c_attn_bias, &c_proj_weight, &c_proj_bias}; }
    std::vector<Tensor*> get_gradients() override { return {&dc_attn_weight, &dc_attn_bias, &dc_proj_weight, &dc_proj_bias}; }
};

} // namespace ns

#endif // NEURAL_SUITE_ATTENTION_H
