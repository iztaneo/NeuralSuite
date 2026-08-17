// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file generate_llm.cpp
 * @brief Autoregressive Text Generation Demo following Google C++ Style Guide.
 */

#include <iostream>
#include <string>
#include <vector>
#include "gpt.h"
#include "tensor.h"
#include "tokenizer.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🤖 Generación Autorregresiva en C++ (Google C++ Style Guide)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  CharTokenizer tokenizer;
  if (!tokenizer.Load("vocab_cpp.txt")) {
    std::cout << "⚠️ No se encontró 'vocab_cpp.txt'. Usando vocabulario por defecto...\n" << std::flush;
    tokenizer.BuildVocab("First Citizen: Speak, speak. MENENIUS: What work's, my countrymen?");
  }

  GPTConfig config;
  config.vocab_size = tokenizer.VocabSize();
  config.block_size = 32;
  config.n_layer = 2;
  config.n_head = 2;
  config.n_embd = 32;

  GPTModel model(config);

  std::string prompt = "First Citizen:\n";
  std::cout << "Prompt de entrada: '" << prompt << "'\n\n" << std::flush;

  std::vector<int> tokens = tokenizer.Encode(prompt);
  int max_new_tokens = 50;

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
    int best_token = 0;
    float max_logit = logits[last_offset];

    for (int v = 1; v < config.vocab_size; ++v) {
      if (logits[last_offset + v] > max_logit) {
        max_logit = logits[last_offset + v];
        best_token = v;
      }
    }

    tokens.push_back(best_token);
  }

  std::string generated = tokenizer.Decode(tokens);
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "Texto Generado por el LLM C++:\n" << generated << "\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;

  return 0;
}
