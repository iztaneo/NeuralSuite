// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file generate_llm.cpp
 * @brief Autoregressive Text Generation Demo matching PyTorch architecture (800K params).
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

int SampleToken(const float* logits, int vocab_size, float temperature = 0.7f, int prev_token = -1, int newline_token = -1) {
  std::vector<float> probs(vocab_size);
  float max_val = logits[0] / temperature;
  for (int v = 1; v < vocab_size; ++v) {
    if (logits[v] / temperature > max_val) max_val = logits[v] / temperature;
  }

  float sum = 0.0f;
  for (int v = 0; v < vocab_size; ++v) {
    probs[v] = std::exp(logits[v] / temperature - max_val);
    if (v == newline_token && prev_token == newline_token) {
      probs[v] *= 0.1f;
    }
    sum += probs[v];
  }

  for (int v = 0; v < vocab_size; ++v) probs[v] /= sum;

  static std::mt19937 rng(std::random_device{}());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng);
}

int main(int argc, char** argv) {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🤖 GENERACIÓN DE TEXTO AUTORREGRESIVA C++ (NeuralSuite 800K)\n" << std::flush;
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
  config.block_size = 64;
  config.n_layer = 4;
  config.n_head = 4;
  config.n_embd = 64;


  GPTModel model(config);
  if (model.LoadWeights("model_cpp.bin")) {
    std::cout << "✅ Pesos del modelo C++ (800K params) cargados desde 'model_cpp.bin'.\n" << std::flush;
  } else {
    std::cout << "⚠️ No se encontró 'model_cpp.bin'. Usando pesos aleatorios...\n" << std::flush;
  }

  std::string prompt = "First Citizen:\n";
  if (argc > 1) {
    prompt = argv[1];
  }

  std::cout << "\nPrompt de entrada: '" << prompt << "'\n" << std::flush;

  std::vector<int> tokens = tokenizer.Encode(prompt);
  int max_new_tokens = 200;
  int newline_token = tokenizer.Encode("\n")[0];

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

    int prev_token = tokens.empty() ? -1 : tokens.back();
    int sampled_token = SampleToken(&logits[last_offset], config.vocab_size, 0.7f, prev_token, newline_token);
    tokens.push_back(sampled_token);
  }

  std::string generated = tokenizer.Decode(tokens);
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "RESULTADO GENERADO POR EL LLM EN C++:\n" << std::flush;
  std::cout << generated << "\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;

  return 0;
}
