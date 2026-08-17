// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file generate_llm.cpp
 * @brief Autoregressive Text Generation Demo loading trained checkpoint model_cpp.bin.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "gpt.h"
#include "tensor.h"
#include "tokenizer.h"

using namespace neuralsuite;

int SampleToken(const float* logits, int vocab_size, float temperature = 0.8f) {
  std::vector<float> probs(vocab_size);
  float max_val = logits[0] / temperature;
  for (int v = 1; v < vocab_size; ++v) {
    if (logits[v] / temperature > max_val) max_val = logits[v] / temperature;
  }

  float sum = 0.0f;
  for (int v = 0; v < vocab_size; ++v) {
    probs[v] = std::exp(logits[v] / temperature - max_val);
    sum += probs[v];
  }

  for (int v = 0; v < vocab_size; ++v) probs[v] /= sum;

  static std::mt19937 rng(42);
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng);
}

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🤖 GENERACIÓN DE TEXTO AUTORREGRESIVA C++ DESDE CHECKPOINT\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  CharTokenizer tokenizer;
  if (!tokenizer.Load("vocab_cpp.txt")) {
    std::cout << "⚠️ No se encontró 'vocab_cpp.txt'. Usando vocabulario por defecto...\n" << std::flush;
    tokenizer.BuildVocab("First Citizen:\nBefore we proceed any further, hear me speak.");
  } else {
    std::cout << "🔤 Vocabulario cargado desde 'vocab_cpp.txt': " << tokenizer.VocabSize() << " caracteres.\n" << std::flush;
  }

  GPTConfig config;
  config.vocab_size = tokenizer.VocabSize();
  config.block_size = 32;
  config.n_layer = 2;
  config.n_head = 2;
  config.n_embd = 32;

  GPTModel model(config);
  if (model.LoadWeights("model_cpp.bin")) {
    std::cout << "✅ Pesos del modelo C++ cargados desde 'model_cpp.bin'.\n" << std::flush;
  } else {
    std::cout << "⚠️ No se encontró 'model_cpp.bin'. Usando pesos no entrenados...\n" << std::flush;
  }

  std::string prompt = "First Citizen:\n";
  std::cout << "\nPrompt de entrada:\n" << prompt << "\n" << std::flush;

  std::vector<int> tokens = tokenizer.Encode(prompt);
  int max_new_tokens = 150;

  for (int step = 0; step < max_new_tokens; ++step) {
    int seq_len = static_cast<int>(tokens.size());
    int start_idx = (seq_len > config.block_size) ? (seq_len - config.block_size) : 0;
    int curr_len = seq_len - start_idx;

    Tensor idx({1, curr_len});
    for (int i = 0; i < curr_len; ++i) {
      idx[i] = static_cast<float>(tokens[start_idx + i]);
    }

    Tensor logits = model.Forward(idx);
    int last_offset = (curr_len - 1) * config.vocab_size;

    int sampled_token = SampleToken(&logits[last_offset], config.vocab_size, 0.8f);
    tokens.push_back(sampled_token);
  }

  std::string generated = tokenizer.Decode(tokens);
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "RESULTADO GENERADO POR EL LLM EN C++:\n" << std::flush;
  std::cout << generated << "\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;

  return 0;
}
