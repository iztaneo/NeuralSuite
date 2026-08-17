// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file attention.h
 * @brief Causal Multi-Head Self-Attention with Full Linear Projection and Exact Backward Pass.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_

#include <cmath>
#include <cstring>
#include <vector>
#include "../layer.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class MultiHeadAttention
 * @brief Multi-Head Causal Self-Attention Layer for Transformer Decoders.
 */
class MultiHeadAttention : public Layer {
 public:
  MultiHeadAttention(int embd, int heads)
      : n_embd_(embd),
        n_head_(heads),
        head_dim_(embd / heads),
        c_attn_(embd, 3 * embd),
        c_proj_(embd, embd) {}

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int seq_len = input.Shape()[1];

    Tensor input_2d({batch_size * seq_len, n_embd_});
    std::memcpy(input_2d.Data(), input.Data(), input.TotalSize() * sizeof(float));

    // 1. Proyección Q, K, V combinada
    qkv_cache_ = c_attn_.Forward(input_2d);

    // 2. Cálculo de Atención Causal por cabeza
    Tensor attn_out({batch_size, seq_len, n_embd_});
    attn_out.Zeros();

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_probs_cache_.Reshape({batch_size, n_head_, seq_len, seq_len});

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < n_head_; ++h) {


        Tensor scores({seq_len, seq_len});
        scores.Zeros();

        for (int i = 0; i < seq_len; ++i) {
          for (int j = 0; j <= i; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim_; ++d) {
              size_t q_idx = (b * seq_len + i) * (3 * n_embd_) + (h * head_dim_ + d);
              size_t k_idx = (b * seq_len + j) * (3 * n_embd_) + n_embd_ + (h * head_dim_ + d);
              dot += qkv_cache_[q_idx] * qkv_cache_[k_idx];
            }
            scores[i * seq_len + j] = dot * scale;
          }
        }

        Tensor attn({seq_len, seq_len});
        CausalSoftmaxForward(scores, attn, seq_len);

        for (int i = 0; i < seq_len; ++i) {
          for (int j = 0; j <= i; ++j) {
            size_t cache_idx = ((b * n_head_ + h) * seq_len + i) * seq_len + j;
            attn_probs_cache_[cache_idx] = attn[i * seq_len + j];
          }
        }

        for (int i = 0; i < seq_len; ++i) {
          for (int d = 0; d < head_dim_; ++d) {
            float val = 0.0f;
            for (int j = 0; j <= i; ++j) {
              size_t v_idx = (b * seq_len + j) * (3 * n_embd_) + 2 * n_embd_ + (h * head_dim_ + d);
              val += attn[i * seq_len + j] * qkv_cache_[v_idx];
            }
            size_t out_idx = (b * seq_len + i) * n_embd_ + (h * head_dim_ + d);
            attn_out[out_idx] = val;
          }
        }
      }
    }

    // 3. Proyección de salida c_proj_
    Tensor attn_out_2d({batch_size * seq_len, n_embd_});
    std::memcpy(attn_out_2d.Data(), attn_out.Data(), attn_out.TotalSize() * sizeof(float));

    Tensor final_2d = c_proj_.Forward(attn_out_2d);
    Tensor final_output({batch_size, seq_len, n_embd_});
    std::memcpy(final_output.Data(), final_2d.Data(), final_2d.TotalSize() * sizeof(float));

    return final_output;
  }

  Tensor Backward(const Tensor& dout) override {
    int batch_size = last_input_.Shape()[0];
    int seq_len = last_input_.Shape()[1];

    // 1. Backward a través de c_proj_
    Tensor dout_2d({batch_size * seq_len, n_embd_});
    std::memcpy(dout_2d.Data(), dout.Data(), dout.TotalSize() * sizeof(float));

    Tensor dattn_head_2d = c_proj_.Backward(dout_2d);
    Tensor dattn_head({batch_size, seq_len, n_embd_});
    std::memcpy(dattn_head.Data(), dattn_head_2d.Data(), dattn_head_2d.TotalSize() * sizeof(float));

    // 2. Backward a través de las cabezas de atención
    Tensor dqkv({batch_size * seq_len, 3 * n_embd_});
    dqkv.Zeros();

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int b = 0; b < batch_size; ++b) {
      for (int h = 0; h < n_head_; ++h) {


        for (int i = 0; i < seq_len; ++i) {
          std::vector<float> dP(i + 1, 0.0f);
          for (int j = 0; j <= i; ++j) {
            float dp = 0.0f;
            for (int d = 0; d < head_dim_; ++d) {
              size_t dout_idx = (b * seq_len + i) * n_embd_ + (h * head_dim_ + d);
              size_t v_idx = (b * seq_len + j) * (3 * n_embd_) + 2 * n_embd_ + (h * head_dim_ + d);
              dp += dattn_head[dout_idx] * qkv_cache_[v_idx];

              size_t cache_idx = ((b * n_head_ + h) * seq_len + i) * seq_len + j;
              dqkv[v_idx] += attn_probs_cache_[cache_idx] * dattn_head[dout_idx];
            }
            dP[j] = dp;
          }

          float sum_dP_P = 0.0f;
          for (int j = 0; j <= i; ++j) {
            size_t cache_idx = ((b * n_head_ + h) * seq_len + i) * seq_len + j;
            sum_dP_P += dP[j] * attn_probs_cache_[cache_idx];
          }

          for (int j = 0; j <= i; ++j) {
            size_t cache_idx = ((b * n_head_ + h) * seq_len + i) * seq_len + j;
            float P_ij = attn_probs_cache_[cache_idx];
            float dS_ij = P_ij * (dP[j] - sum_dP_P) * scale;

            for (int d = 0; d < head_dim_; ++d) {
              size_t q_idx = (b * seq_len + i) * (3 * n_embd_) + (h * head_dim_ + d);
              size_t k_idx = (b * seq_len + j) * (3 * n_embd_) + n_embd_ + (h * head_dim_ + d);

              dqkv[q_idx] += dS_ij * qkv_cache_[k_idx];
              dqkv[k_idx] += dS_ij * qkv_cache_[q_idx];
            }
          }
        }
      }
    }

    // 3. Backward a través de c_attn_
    Tensor dx_2d = c_attn_.Backward(dqkv);
    Tensor dx({batch_size, seq_len, n_embd_});
    std::memcpy(dx.Data(), dx_2d.Data(), dx.TotalSize() * sizeof(float));

    return dx;
  }

  std::vector<Tensor*> GetParameters() override {
    std::vector<Tensor*> p = c_attn_.GetParameters();
    auto p2 = c_proj_.GetParameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
  }

  static void ApplyRoPE(float* vec, int head_dim, int pos) {
    for (int i = 0; i < head_dim - 1; i += 2) {
      float theta = static_cast<float>(pos) / std::pow(10000.0f, static_cast<float>(i) / head_dim);
      float cos_th = std::cos(theta);
      float sin_th = std::sin(theta);
      float x0 = vec[i];
      float x1 = vec[i + 1];
      vec[i]     = x0 * cos_th - x1 * sin_th;
      vec[i + 1] = x0 * sin_th + x1 * cos_th;
    }
  }

  void ClearKVCache() {

    k_cache_.clear();
    v_cache_.clear();
  }

  Tensor ForwardWithKVCache(const Tensor& single_token_input) {
    // single_token_input shape: [1, 1, n_embd]
    int batch_size = 1;
    int seq_len = 1;

    Tensor input_2d({1, n_embd_});
    std::memcpy(input_2d.Data(), single_token_input.Data(), n_embd_ * sizeof(float));

    Tensor qkv = c_attn_.Forward(input_2d);

    // Extraer Q_t, K_t, V_t para el token actual
    std::vector<float> q_curr(n_embd_), k_curr(n_embd_), v_curr(n_embd_);
    std::memcpy(q_curr.data(), qkv.Data(), n_embd_ * sizeof(float));
    std::memcpy(k_curr.data(), qkv.Data() + n_embd_, n_embd_ * sizeof(float));
    std::memcpy(v_curr.data(), qkv.Data() + 2 * n_embd_, n_embd_ * sizeof(float));

    // Acumular K y V en la memoria caché
    k_cache_.push_back(k_curr);
    v_cache_.push_back(v_curr);

    int total_t = static_cast<int>(k_cache_.size());

    Tensor attn_out({1, 1, n_embd_});
    attn_out.Zeros();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int h = 0; h < n_head_; ++h) {
      std::vector<float> scores(total_t, 0.0f);
      float max_score = -1e9f;

      for (int t = 0; t < total_t; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim_; ++d) {
          float q_val = q_curr[h * head_dim_ + d];
          float k_val = k_cache_[t][h * head_dim_ + d];
          dot += q_val * k_val;
        }
        scores[t] = dot * scale;
        if (scores[t] > max_score) max_score = scores[t];
      }

      float sum_exp = 0.0f;
      for (int t = 0; t < total_t; ++t) {
        scores[t] = std::exp(scores[t] - max_score);
        sum_exp += scores[t];
      }

      for (int d = 0; d < head_dim_; ++d) {
        float val = 0.0f;
        for (int t = 0; t < total_t; ++t) {
          float prob = scores[t] / sum_exp;
          val += prob * v_cache_[t][h * head_dim_ + d];
        }
        attn_out[h * head_dim_ + d] = val;
      }
    }

    Tensor attn_out_2d({1, n_embd_});
    std::memcpy(attn_out_2d.Data(), attn_out.Data(), n_embd_ * sizeof(float));

    Tensor final_2d = c_proj_.Forward(attn_out_2d);
    Tensor final_output({1, 1, n_embd_});
    std::memcpy(final_output.Data(), final_2d.Data(), n_embd_ * sizeof(float));

    return final_output;
  }

 private:
  int n_embd_;
  int n_head_;
  int head_dim_;

  Linear c_attn_;
  Linear c_proj_;

  Tensor last_input_;
  Tensor qkv_cache_;
  Tensor attn_probs_cache_;

  std::vector<std::vector<float>> k_cache_;
  std::vector<std::vector<float>> v_cache_;
};


}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
