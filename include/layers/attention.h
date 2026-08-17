// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file attention.h
 * @brief Causal Multi-Head Self-Attention following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_

#include <vector>
#include "../layer.h"

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
        c_attn_weight_({3 * embd, embd}),
        c_attn_bias_({3 * embd}),
        c_proj_weight_({embd, embd}),
        c_proj_bias_({embd}),
        dc_attn_weight_({3 * embd, embd}),
        dc_attn_bias_({3 * embd}),
        dc_proj_weight_({embd, embd}),
        dc_proj_bias_({embd}) {
    c_attn_weight_.XavierInit(embd, 3 * embd);
    c_proj_weight_.XavierInit(embd, embd);
    c_attn_bias_.Zeros();
    c_proj_bias_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int seq_len = input.Shape()[1];

    Tensor qkv({batch_size * seq_len, 3 * n_embd_});
    Tensor input_2d({batch_size * seq_len, n_embd_});
    std::memcpy(input_2d.Data(), input.Data(), input.TotalSize() * sizeof(float));

    Tensor weight_t = Transpose(c_attn_weight_);
    MatMul(input_2d, weight_t, qkv);

    Tensor output({batch_size, seq_len, n_embd_});
    output.Zeros();

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

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
              dot += qkv[q_idx] * qkv[k_idx];
            }
            scores[i * seq_len + j] = dot * scale;
          }
        }

        Tensor attn({seq_len, seq_len});
        CausalSoftmaxForward(scores, attn, seq_len);

        for (int i = 0; i < seq_len; ++i) {
          for (int d = 0; d < head_dim_; ++d) {
            float val = 0.0f;
            for (int j = 0; j <= i; ++j) {
              size_t v_idx = (b * seq_len + j) * (3 * n_embd_) + 2 * n_embd_ + (h * head_dim_ + d);
              val += attn[i * seq_len + j] * qkv[v_idx];
            }
            size_t out_idx = (b * seq_len + i) * n_embd_ + (h * head_dim_ + d);
            output[out_idx] = val;
          }
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    Tensor dx(last_input_.Shape());
    dx.Zeros();
    dc_attn_weight_.Zeros(); dc_attn_bias_.Zeros();
    dc_proj_weight_.Zeros(); dc_proj_bias_.Zeros();
    return dx;
  }

  std::vector<Tensor*> GetParameters() override {
    return {&c_attn_weight_, &c_attn_bias_, &c_proj_weight_, &c_proj_bias_};
  }

  std::vector<Tensor*> GetGradients() override {
    return {&dc_attn_weight_, &dc_attn_bias_, &dc_proj_weight_, &dc_proj_bias_};
  }

 private:
  int n_embd_;
  int n_head_;
  int head_dim_;

  Tensor c_attn_weight_;
  Tensor c_attn_bias_;
  Tensor c_proj_weight_;
  Tensor c_proj_bias_;

  Tensor dc_attn_weight_;
  Tensor dc_attn_bias_;
  Tensor dc_proj_weight_;
  Tensor dc_proj_bias_;

  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
