// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/attention.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/attention.h"

namespace neuralsuite {

Tensor MultiHeadAttentionReference::Forward(const Tensor& input) {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int seq_len = input.Shape()[1];

    // Vista: aplanar [B, T, C] a [B*T, C] no requiere copiar nada.
    const Tensor input_2d = input.View({batch_size * seq_len, n_embd_});

    // 1. Proyección Q, K, V combinada
    qkv_cache_ = c_attn_.Forward(input_2d);

    // 2. Cálculo de Atención Causal por cabeza
    Tensor attn_out({batch_size, seq_len, n_embd_});
    attn_out.Zeros();

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_probs_cache_.Resize({batch_size, n_head_, seq_len, seq_len});

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
    const Tensor attn_out_2d = attn_out.View({batch_size * seq_len, n_embd_});

    Tensor final_2d = c_proj_.Forward(attn_out_2d);
    final_2d.Reshape({batch_size, seq_len, n_embd_});
    return final_2d;
  }

Tensor MultiHeadAttentionReference::Backward(const Tensor& dout) {
    int batch_size = last_input_.Shape()[0];
    int seq_len = last_input_.Shape()[1];

    // 1. Backward a través de c_proj_
    const Tensor dout_2d = dout.View({batch_size * seq_len, n_embd_});

    Tensor dattn_head = c_proj_.Backward(dout_2d);
    dattn_head.Reshape({batch_size, seq_len, n_embd_});

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
    dx_2d.Reshape({batch_size, seq_len, n_embd_});
    return dx_2d;
  }

void MultiHeadAttentionReference::ClearKVCache() {

    k_cache_.clear();
    v_cache_.clear();
  }

Tensor MultiHeadAttentionReference::ForwardWithKVCache(const Tensor& single_token_input) {
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

Tensor MultiHeadAttention::Forward(const Tensor& input) {
    last_input_ = input;
    const int B = input.Shape()[0];
    const int T = input.Shape()[1];
    const int C = n_embd_, D = head_dim_;

    const Tensor input_2d = input.View({B * T, C});
    qkv_cache_ = c_attn_.Forward(input_2d);

    Tensor attn_out({B, T, C});
    attn_out.Zeros();
    attn_probs_cache_.Resize({B, n_head_, T, T});
    const float escala = 1.0f / std::sqrt(static_cast<float>(D));

    parallel::ParallelFor(B * n_head_, /*min_per_thread=*/1, [&](int desde, int hasta) {
      Tensor Q({T, D}), Kt({D, T}), V({T, D});
      Tensor puntuaciones, P, salida;

      for (int u = desde; u < hasta; ++u) {
        const int b = u / n_head_, h = u % n_head_;
        Extraer(b, h, T, &Q, &Kt, &V);

        MatMul(Q, Kt, puntuaciones);                       // [T, T]
        for (size_t i = 0; i < puntuaciones.TotalSize(); ++i) puntuaciones[i] *= escala;
        CausalSoftmaxForward(puntuaciones, P, T);
        MatMul(P, V, salida);                              // [T, D]

        std::memcpy(attn_probs_cache_.Data() + static_cast<size_t>(u) * T * T, P.Data(),
                    static_cast<size_t>(T) * T * sizeof(float));
        for (int i = 0; i < T; ++i) {
          std::memcpy(attn_out.Data() + (static_cast<size_t>(b) * T + i) * C + h * D,
                      salida.Data() + static_cast<size_t>(i) * D, D * sizeof(float));
        }
      }
    });

    const Tensor attn_out_2d = attn_out.View({B * T, C});
    Tensor final_2d = c_proj_.Forward(attn_out_2d);
    final_2d.Reshape({B, T, C});
    return final_2d;
  }

Tensor MultiHeadAttention::Backward(const Tensor& dout) {
    const int B = last_input_.Shape()[0];
    const int T = last_input_.Shape()[1];
    const int C = n_embd_, D = head_dim_;

    const Tensor dout_2d = dout.View({B * T, C});
    Tensor dattn = c_proj_.Backward(dout_2d);
    dattn.Reshape({B, T, C});

    Tensor dqkv({B * T, 3 * C});
    dqkv.Zeros();
    const float escala = 1.0f / std::sqrt(static_cast<float>(D));

    parallel::ParallelFor(B * n_head_, /*min_per_thread=*/1, [&](int desde, int hasta) {
      Tensor Q({T, D}), Kt({D, T}), V({T, D}), K({T, D});
      Tensor dSalida({T, D}), Vt({D, T}), dP, dV, dS({T, T}), Pt({T, T}), dQ, dK, dSt({T, T});

      for (int u = desde; u < hasta; ++u) {
        const int b = u / n_head_, h = u % n_head_;
        Extraer(b, h, T, &Q, &Kt, &V);
        for (int i = 0; i < T; ++i) {
          for (int d = 0; d < D; ++d) K[static_cast<size_t>(i) * D + d] = Kt[static_cast<size_t>(d) * T + i];
        }
        for (int i = 0; i < T; ++i) {
          std::memcpy(dSalida.Data() + static_cast<size_t>(i) * D,
                      dattn.Data() + (static_cast<size_t>(b) * T + i) * C + h * D,
                      D * sizeof(float));
        }

        const float* P = attn_probs_cache_.Data() + static_cast<size_t>(u) * T * T;
        for (int i = 0; i < T; ++i) {
          for (int d = 0; d < D; ++d) Vt[static_cast<size_t>(d) * T + i] = V[static_cast<size_t>(i) * D + d];
          for (int j = 0; j < T; ++j) Pt[static_cast<size_t>(j) * T + i] = P[static_cast<size_t>(i) * T + j];
        }

        MatMul(dSalida, Vt, dP);      // [T, T]
        MatMul(Pt, dSalida, dV);      // [T, D]

        // dS = P * (dP - suma_fila(dP * P)) * escala. Donde P vale cero -el
        // triangulo superior que enmascaro el softmax- dS sale cero solo.
        for (int i = 0; i < T; ++i) {
          double suma = 0.0;
          for (int j = 0; j <= i; ++j) {
            suma += static_cast<double>(dP[static_cast<size_t>(i) * T + j]) * P[static_cast<size_t>(i) * T + j];
          }
          for (int j = 0; j < T; ++j) {
            const float p = P[static_cast<size_t>(i) * T + j];
            dS[static_cast<size_t>(i) * T + j] =
                (j <= i) ? p * (dP[static_cast<size_t>(i) * T + j] - static_cast<float>(suma)) * escala
                         : 0.0f;
          }
        }
        for (int i = 0; i < T; ++i) {
          for (int j = 0; j < T; ++j) dSt[static_cast<size_t>(j) * T + i] = dS[static_cast<size_t>(i) * T + j];
        }

        MatMul(dS, K, dQ);            // [T, D]
        MatMul(dSt, Q, dK);           // [T, D]

        // Cada cabeza escribe en su propio tramo de columnas: sin solape.
        for (int i = 0; i < T; ++i) {
          const size_t fila = (static_cast<size_t>(b) * T + i) * 3 * C;
          std::memcpy(dqkv.Data() + fila + h * D, dQ.Data() + static_cast<size_t>(i) * D,
                      D * sizeof(float));
          std::memcpy(dqkv.Data() + fila + C + h * D, dK.Data() + static_cast<size_t>(i) * D,
                      D * sizeof(float));
          std::memcpy(dqkv.Data() + fila + 2 * C + h * D, dV.Data() + static_cast<size_t>(i) * D,
                      D * sizeof(float));
        }
      }
    });

    Tensor dx_2d = c_attn_.Backward(dqkv);
    dx_2d.Reshape({B, T, C});
    return dx_2d;
  }

Tensor MultiHeadAttention::ForwardWithKVCache(const Tensor& single_token_input) {
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

void MultiHeadAttention::ClearKVCache() {
    k_cache_.clear();
    v_cache_.clear();
  }

void MultiHeadAttention::Extraer(int b, int h, int T, Tensor* Q, Tensor* Kt, Tensor* V) const {
    const int C = n_embd_, D = head_dim_;
    for (int i = 0; i < T; ++i) {
      const size_t fila = (static_cast<size_t>(b) * T + i) * 3 * C;
      std::memcpy(Q->Data() + static_cast<size_t>(i) * D, qkv_cache_.Data() + fila + h * D,
                  D * sizeof(float));
      std::memcpy(V->Data() + static_cast<size_t>(i) * D,
                  qkv_cache_.Data() + fila + 2 * C + h * D, D * sizeof(float));
      for (int d = 0; d < D; ++d) {
        (*Kt)[static_cast<size_t>(d) * T + i] = qkv_cache_[fila + C + h * D + d];
      }
    }
  }

}  // namespace neuralsuite
