// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file gpt.h
 * @brief Full Decoder-Only Transformer (GPT) Model following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_GPT_H_
#define NEURAL_SUITE_INCLUDE_GPT_H_

#include <memory>
#include <vector>
#include "activations.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"
#include "layers/linear.h"
#include "losses.h"
#include "tensor.h"
#include "tokenizer.h"

namespace neuralsuite {

/**
 * @struct GPTConfig
 * @brief Hyperparameter configuration for GPT model architecture.
 */
struct GPTConfig {
  int vocab_size = 54;
  int block_size = 64;
  int n_layer = 4;
  int n_head = 4;
  int n_embd = 128;
};

/**
 * @class GPTBlock
 * @brief Transformer Decoder Block with Pre-LN Residual Connections.
 */
class GPTBlock : public Layer {
 public:
  explicit GPTBlock(const GPTConfig& config)
      : ln_1_(config.n_embd),
        attn_(config.n_embd, config.n_head),
        ln_2_(config.n_embd),
        mlp_fc_(config.n_embd, 4 * config.n_embd),
        mlp_gelu_(ActivationType::kGelu),
        mlp_proj_(4 * config.n_embd, config.n_embd) {}

  Tensor Forward(const Tensor& input) override {
    Tensor x_norm1 = ln_1_.Forward(input);
    Tensor attn_out = attn_.Forward(x_norm1);
    Tensor x1(input.Shape());
    ElementwiseAdd(input, attn_out, x1);

    Tensor x_norm2 = ln_2_.Forward(x1);
    Tensor fc_out = mlp_fc_.Forward(x_norm2);
    Tensor gelu_out = mlp_gelu_.Forward(fc_out);
    Tensor proj_out = mlp_proj_.Forward(gelu_out);
    Tensor x2(x1.Shape());
    ElementwiseAdd(x1, proj_out, x2);

    return x2;
  }

  Tensor Backward(const Tensor& dout) override {
    // 1. Backward del bloque MLP con conexión residual
    Tensor dproj_out = mlp_proj_.Backward(dout);
    Tensor dgelu_out = mlp_gelu_.Backward(dproj_out);
    Tensor dfc_out = mlp_fc_.Backward(dgelu_out);
    Tensor dx_norm2 = ln_2_.Backward(dfc_out);

    Tensor dx1(dout.Shape());
    ElementwiseAdd(dout, dx_norm2, dx1);

    // 2. Backward de la Atención Multi-Cabeza con conexión residual
    Tensor dattn_out = attn_.Backward(dx1);
    Tensor dx_norm1 = ln_1_.Backward(dattn_out);

    Tensor dx(dx1.Shape());
    ElementwiseAdd(dx1, dx_norm1, dx);

    return dx;
  }


  Tensor ForwardWithKVCache(const Tensor& input) {
    Tensor x_norm1 = ln_1_.Forward(input);
    Tensor attn_out = attn_.ForwardWithKVCache(x_norm1);
    Tensor x1(input.Shape());
    ElementwiseAdd(input, attn_out, x1);

    Tensor x_norm2 = ln_2_.Forward(x1);
    Tensor fc_out = mlp_fc_.Forward(x_norm2);
    Tensor gelu_out = mlp_gelu_.Forward(fc_out);
    Tensor proj_out = mlp_proj_.Forward(gelu_out);
    Tensor x2(x1.Shape());
    ElementwiseAdd(x1, proj_out, x2);

    return x2;
  }

  void ClearKVCache() {
    attn_.ClearKVCache();
  }

  std::vector<Tensor*> GetParameters() override {

    std::vector<Tensor*> p;
    for (auto p1 : ln_1_.GetParameters()) p.push_back(p1);
    for (auto p2 : attn_.GetParameters()) p.push_back(p2);
    for (auto p3 : ln_2_.GetParameters()) p.push_back(p3);
    for (auto p4 : mlp_fc_.GetParameters()) p.push_back(p4);
    for (auto p5 : mlp_proj_.GetParameters()) p.push_back(p5);
    return p;
  }

  std::vector<Tensor*> GetGradients() override {
    std::vector<Tensor*> g;
    for (auto g1 : ln_1_.GetGradients()) g.push_back(g1);
    for (auto g2 : attn_.GetGradients()) g.push_back(g2);
    for (auto g3 : ln_2_.GetGradients()) g.push_back(g3);
    for (auto g4 : mlp_fc_.GetGradients()) g.push_back(g4);
    for (auto g5 : mlp_proj_.GetGradients()) g.push_back(g5);
    return g;
  }

 private:
  LayerNormLayer ln_1_;
  MultiHeadAttention attn_;
  LayerNormLayer ln_2_;
  Linear mlp_fc_;
  Activation mlp_gelu_;
  Linear mlp_proj_;
};

/**
 * @class GPTModel
 * @brief Complete GPT LLM Model.
 */
class GPTModel {
 public:
  explicit GPTModel(const GPTConfig& cfg)
      : config_(cfg),
        wte_(cfg.vocab_size, cfg.n_embd),
        wpe_(cfg.block_size, cfg.n_embd),
        ln_f_(cfg.n_embd) {
    for (int i = 0; i < cfg.n_layer; ++i) {
      blocks_.push_back(std::make_shared<GPTBlock>(cfg));
    }
  }


  void ClearKVCache() {
    for (auto& block : blocks_) {
      block->ClearKVCache();
    }
  }

  Tensor ForwardWithKVCache(int token_idx, int pos_idx) {
    Tensor tok_tensor({1, 1});
    tok_tensor[0] = static_cast<float>(token_idx);
    Tensor tok_emb = wte_.Forward(tok_tensor);

    Tensor pos_tensor({1, 1});
    pos_tensor[0] = static_cast<float>(pos_idx);
    Tensor pos_emb = wpe_.Forward(pos_tensor);

    Tensor x({1, 1, config_.n_embd});
    for (int d = 0; d < config_.n_embd; ++d) {
      x[d] = tok_emb[d] + pos_emb[d];
    }

    for (auto& block : blocks_) {
      x = block->ForwardWithKVCache(x);
    }

    x = ln_f_.Forward(x);

    Tensor x_2d({1, config_.n_embd});
    std::memcpy(x_2d.Data(), x.Data(), config_.n_embd * sizeof(float));

    Tensor wte_T = Transpose(wte_.Weight());
    Tensor logits_2d;
    MatMul(x_2d, wte_T, logits_2d);

    Tensor logits({1, 1, config_.vocab_size});
    std::memcpy(logits.Data(), logits_2d.Data(), config_.vocab_size * sizeof(float));

    return logits;
  }

  Tensor Forward(const Tensor& idx) {

    int batch_size = idx.Shape()[0];
    int seq_len = idx.Shape()[1];

    last_x_2d_.Reshape({batch_size * seq_len, config_.n_embd});

    Tensor tok_emb = wte_.Forward(idx);

    Tensor pos_idx({1, seq_len});
    for (int t = 0; t < seq_len; ++t) pos_idx[t] = static_cast<float>(t);
    Tensor pos_emb = wpe_.Forward(pos_idx);

    Tensor x({batch_size, seq_len, config_.n_embd});
    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        for (int d = 0; d < config_.n_embd; ++d) {
          size_t idx_3d = (b * seq_len + t) * config_.n_embd + d;
          size_t pos_3d = t * config_.n_embd + d;
          x[idx_3d] = tok_emb[idx_3d] + pos_emb[pos_3d];
        }
      }
    }

    for (auto& block : blocks_) {
      x = block->Forward(x);
    }

    x = ln_f_.Forward(x);

    std::memcpy(last_x_2d_.Data(), x.Data(), x.TotalSize() * sizeof(float));

    // Weight Tying: logits = x_2d * W_wte^T
    Tensor wte_T = Transpose(wte_.Weight());
    Tensor logits_2d;
    MatMul(last_x_2d_, wte_T, logits_2d);

    Tensor logits({batch_size, seq_len, config_.vocab_size});
    std::memcpy(logits.Data(), logits_2d.Data(), logits_2d.TotalSize() * sizeof(float));

    return logits;
  }

  Tensor Backward(const Tensor& dlogits) {
    int batch_size = dlogits.Shape()[0];
    int seq_len = dlogits.Shape()[1];

    Tensor dlogits_2d({batch_size * seq_len, config_.vocab_size});
    std::memcpy(dlogits_2d.Data(), dlogits.Data(), dlogits.TotalSize() * sizeof(float));

    // Weight Tying Backward:
    // dx_2d = dlogits_2d * W_wte
    Tensor dx_2d;
    MatMul(dlogits_2d, wte_.Weight(), dx_2d);

    // dW_wte += dlogits_2d^T * x_2d
    Tensor dlogits_2d_T = Transpose(dlogits_2d);
    Tensor dwte;
    MatMul(dlogits_2d_T, last_x_2d_, dwte);

    Tensor* wte_grad = wte_.GetGradients()[0];
    for (size_t i = 0; i < wte_grad->TotalSize(); ++i) {
      (*wte_grad)[i] += dwte[i];
    }


    Tensor dx({batch_size, seq_len, config_.n_embd});
    std::memcpy(dx.Data(), dx_2d.Data(), dx_2d.TotalSize() * sizeof(float));

    dx = ln_f_.Backward(dx);

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
      dx = (*it)->Backward(dx);
    }

    wte_.Backward(dx);
    wpe_.Backward(dx);

    return dx;
  }

  std::vector<Tensor*> GetParameters() {

    std::vector<Tensor*> p;
    for (auto p1 : wte_.GetParameters()) p.push_back(p1);
    for (auto p2 : wpe_.GetParameters()) p.push_back(p2);
    for (auto& block : blocks_) {
      for (auto pb : block->GetParameters()) p.push_back(pb);
    }
    for (auto pf : ln_f_.GetParameters()) p.push_back(pf);
    return p;
  }

  std::vector<Tensor*> GetGradients() {
    std::vector<Tensor*> g;
    for (auto g1 : wte_.GetGradients()) g.push_back(g1);
    for (auto g2 : wpe_.GetGradients()) g.push_back(g2);
    for (auto& block : blocks_) {
      for (auto gb : block->GetGradients()) g.push_back(gb);
    }
    for (auto gf : ln_f_.GetGradients()) g.push_back(gf);
    return g;
  }

  bool SaveWeights(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    for (auto p : GetParameters()) {
      out.write(reinterpret_cast<const char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

  bool LoadWeights(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;
    for (auto p : GetParameters()) {
      in.read(reinterpret_cast<char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

  [[nodiscard]] const GPTConfig& Config() const { return config_; }

 private:
  GPTConfig config_;
  Embedding wte_;
  Embedding wpe_;
  std::vector<std::shared_ptr<GPTBlock>> blocks_;
  LayerNormLayer ln_f_;
  Tensor last_x_2d_;
};


}  // namespace neuralsuite


#endif  // NEURAL_SUITE_INCLUDE_GPT_H_
