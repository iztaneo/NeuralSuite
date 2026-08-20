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
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parallel.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class MultiHeadAttentionReference
 * @brief La atencion escrita como su definicion, elemento a elemento.
 *
 * Se conserva como oraculo, igual que Conv2DReference y LSTMReference. Es lenta
 * -1.3 GFLOP/s en un solo hilo- pero se lee al lado de las formulas y no tiene
 * ninguna reordenacion de memoria que pueda desalinearse.
 *
 * @class MultiHeadAttention
 * @brief Multi-Head Causal Self-Attention Layer for Transformer Decoders.
 *
 * La posicion llega por embeddings aprendidos (`wpe_` del modelo), no por
 * rotacion de Q y K. Existio aqui un `ApplyRoPE()` que ningun forward llamaba:
 * sugeria una capacidad que el modelo no tiene, asi que se retiro. RoPE sigue
 * planteado en docs/FUTURE_PLAN_KVCACHE_ROPE.md, y anadirlo exige rotar Q y K
 * en el forward, propagar por esa rotacion en el backward con su gradient
 * check, y actualizar la implementacion de referencia en PyTorch para que la
 * comparacion siga siendo valida.
 */
class MultiHeadAttentionReference : public Layer {
 public:
  MultiHeadAttentionReference(int embd, int heads)
      : n_embd_(embd),
        n_head_(heads),
        head_dim_(embd / heads),
        // 0.02 es la convencion de GPT-2 para todas sus densas. Al pasar Linear
        // a Xavier por defecto se actualizo gpt.h y se olvido este archivo, de
        // modo que la atencion quedo con una inicializacion distinta del resto
        // del modelo sin que nada lo senalara.
        c_attn_(embd, 3 * embd, /*init_std=*/0.02f),
        c_proj_(embd, embd, /*init_std=*/0.02f) {
    Register(&c_attn_, "c_attn");
    Register(&c_proj_, "c_proj");
    // Si embd no es múltiplo de heads, head_dim_ trunca y las dimensiones
    // sobrantes quedan sin asignar a ninguna cabeza, en silencio.
    if (embd <= 0 || heads <= 0) {
      throw std::invalid_argument("MultiHeadAttentionReference: embd y heads deben ser positivos.");
    }
    if (embd % heads != 0) {
      throw std::invalid_argument(
          "MultiHeadAttention: n_embd (" + std::to_string(embd) +
          ") debe ser multiplo de n_head (" + std::to_string(heads) + ").");
    }
  }

  Tensor Forward(const Tensor& input) override {
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

  Tensor Backward(const Tensor& dout) override {
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


/**
 * @class MultiHeadAttention
 * @brief La misma atencion, resuelta con multiplicaciones de matrices.
 *
 * Dentro de la atencion hay dos productos de matrices por cada cabeza:
 *
 *     puntuaciones = Q · K^T        salida = P · V
 *
 * y en `MultiHeadAttentionReference` estan escritos como bucles que recorren
 * cada elemento. Eso daba 1.3 GFLOP/s teniendo un `MatMul` que alcanza 232 y
 * diez nucleos sin usar. Es el tercer y ultimo caso del mismo patron, despues
 * de la convolucion y la celda recurrente.
 *
 * El coste crece con el cuadrado de la longitud del contexto, y eso **no se
 * arregla**: cada posicion mira a todas las anteriores, esa es la definicion
 * del mecanismo. Lo que se arregla es la constante. Medido sobre el GPT de
 * `train_llm`, pasar de contexto 64 a 256 multiplicaba el tiempo por trece
 * cuando lo inevitable era cuatro.
 *
 * La reordenacion que hace falta: `c_attn_` entrega Q, K y V entrelazados en
 * una sola fila de `3 * n_embd`, y cada cabeza ocupa un tramo de esa fila. Para
 * multiplicar hay que extraer cada cabeza como una matriz contigua `[T,
 * head_dim]`. Es el mismo trabajo que hace `im2col` en la convolucion: copiar
 * para poder leer seguido.
 *
 * Se calcula el cuadrado completo de puntuaciones, incluida la mitad que la
 * mascara causal descarta. Es el doble de operaciones que el bucle triangular
 * del oraculo, y aun asi sale mucho mas rapido: un `MatMul` sobre memoria
 * contigua rinde mas que la mitad de las operaciones hechas a saltos.
 * `CausalSoftmaxForward` pone a cero el triangulo superior, de modo que la
 * mascara se aplica sola y el gradiente hereda esos ceros sin tener que
 * enmascararlo aparte.
 *
 * El reparto entre hilos va por pareja (muestra, cabeza). Son independientes y
 * -esto es lo que permite hacerlo sin reduccion- cada una escribe en un tramo
 * distinto de `dqkv`: la cabeza `h` solo toca las columnas `h*head_dim` en
 * adelante. No hay dos hilos que sumen sobre la misma posicion, asi que el
 * resultado es identico con uno o con diez.
 */
class MultiHeadAttention : public Layer {
 public:
  MultiHeadAttention(int n_embd, int n_head)
      : n_embd_(n_embd),
        n_head_(n_head),
        head_dim_(n_embd / n_head),
        c_attn_(n_embd, 3 * n_embd, /*init_std=*/0.02f),
        c_proj_(n_embd, n_embd, /*init_std=*/0.02f) {
    if (n_head <= 0 || n_embd <= 0) {
      throw std::invalid_argument("MultiHeadAttention: n_embd y n_head deben ser positivos.");
    }
    if (n_embd % n_head != 0) {
      throw std::invalid_argument(
          "MultiHeadAttention: n_embd (" + std::to_string(n_embd) +
          ") debe ser divisible entre n_head (" + std::to_string(n_head) + ").");
    }
    Register(&c_attn_, "c_attn");
    Register(&c_proj_, "c_proj");
  }

  Tensor Forward(const Tensor& input) override {
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

  Tensor Backward(const Tensor& dout) override {
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

  [[nodiscard]] const Linear& AttnProjection() const { return c_attn_; }
  Linear& AttnProjection() { return c_attn_; }
  [[nodiscard]] const Linear& OutProjection() const { return c_proj_; }
  Linear& OutProjection() { return c_proj_; }

  /**
   * @brief Un solo token, reutilizando las claves y valores ya calculados.
   *
   * Este camino se queda escalar a proposito: procesa **un** token, de modo que
   * no hay matrices que multiplicar y reordenar la memoria costaria mas que el
   * propio calculo. Es el mismo codigo que la version de referencia.
   */
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

  void ClearKVCache() {
    k_cache_.clear();
    v_cache_.clear();
  }


 private:
  /**
   * @brief Saca Q, K^T y V de una cabeza como matrices contiguas.
   *
   * `c_attn_` los entrega entrelazados —cada fila lleva los tres, y dentro de
   * cada uno un tramo por cabeza—, que es una disposicion imposible de
   * multiplicar sin saltar por la memoria. K sale ya transpuesta porque asi es
   * como entra en el producto.
   */
  void Extraer(int b, int h, int T, Tensor* Q, Tensor* Kt, Tensor* V) const {
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
