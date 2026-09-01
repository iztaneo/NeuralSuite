// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file gpt.h
 * @brief Full Decoder-Only Transformer (GPT) Model following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_GPT_H_
#define NEURAL_SUITE_INCLUDE_GPT_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "activations.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"
#include "layers/linear.h"
#include "losses.h"
#include "serialization.h"
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
        // 0.02 es la convencion de GPT-2 para todas sus densas, y se mantiene
        // aqui de forma explicita: desde que `Linear` usa Xavier por defecto,
        // dejarlo implicito cambiaria la inicializacion del modelo.
        mlp_fc_(config.n_embd, 4 * config.n_embd, /*init_std=*/0.02f),
        mlp_gelu_(ActivationType::kGelu),
        mlp_proj_(4 * config.n_embd, config.n_embd, /*init_std=*/0.02f) {
    // El orden de registro fija el orden de los parametros. Antes esta misma
    // secuencia se repetia a mano en dos metodos que debian coincidir entre si.
    Register(&ln_1_, "ln_1");
    Register(&attn_, "attn");
    Register(&ln_2_, "ln_2");
    Register(&mlp_fc_, "mlp_fc");
    Register(&mlp_proj_, "mlp_proj");
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;


  Tensor ForwardWithKVCache(const Tensor& input);

  void ClearKVCache() {
    attn_.ClearKVCache();
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
class GPTModel : public Module {
 public:
  explicit GPTModel(const GPTConfig& cfg)
      : config_(cfg),
        wte_(cfg.vocab_size, cfg.n_embd),
        wpe_(cfg.block_size, cfg.n_embd),
        ln_f_(cfg.n_embd) {
    Register(&wte_, "wte");
    Register(&wpe_, "wpe");
    for (int i = 0; i < cfg.n_layer; ++i) {
      blocks_.push_back(std::make_shared<GPTBlock>(cfg));
      Register(blocks_.back().get(), "blocks." + std::to_string(i));
    }
    Register(&ln_f_, "ln_f");
  }


  void ClearKVCache();

  Tensor ForwardWithKVCache(int token_idx, int pos_idx);

  Tensor Forward(const Tensor& idx);

  Tensor Backward(const Tensor& dlogits);

  /** @brief Pesos y gradientes, derivados ambos del recorrido de submodulos. */
  [[nodiscard]] std::vector<Tensor*> GetParameters();

  [[nodiscard]] std::vector<Tensor*> GetGradients();

  /**
   * @brief Guarda los pesos en formato NSF, con la arquitectura declarada.
   *
   * Los metadatos permiten que la carga rechace un archivo que no corresponda a
   * esta configuracion en vez de aceptar cualquier secuencia de floats.
   */
  bool SaveWeights(const std::string& filepath);

  /** @brief Carga pesos NSF, comprobando que correspondan a esta arquitectura. */
  bool LoadWeights(const std::string& filepath);

  /** @brief Descripcion de la arquitectura que viaja dentro del archivo. */
  [[nodiscard]] std::map<std::string, std::string> ArchitectureMetadata() const;

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
