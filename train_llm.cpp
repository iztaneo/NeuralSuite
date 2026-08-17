// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file train_llm.cpp
 * @brief Transformer LLM Training Script following Google C++ Style Guide.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "gpt.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"
#include "tokenizer.h"

using namespace neuralsuite;

int main(int argc, char** argv) {
  std::string data_path = "sample_data/input.txt";
  if (argc > 1) {
    data_path = argv[1];
  } else if (!std::ifstream(data_path).good() && std::ifstream("../sample_data/input.txt").good()) {
    data_path = "../sample_data/input.txt";
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "🚀 Entrenamiento de LLM en C++ (Google C++ Style Guide)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  std::ifstream file(data_path);
  if (!file.is_open()) {
    std::cerr << "❌ No se pudo abrir el archivo de datos: " << data_path << "\n" << std::flush;
    return 1;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string text = buffer.str();
  std::cout << "📄 Dataset cargado: " << text.size() << " caracteres.\n" << std::flush;

  CharTokenizer tokenizer(text);
  std::cout << "🔤 Vocabulario del tokenizador: " << tokenizer.VocabSize() << " caracteres únicos.\n" << std::flush;
  tokenizer.Save("vocab_cpp.txt");

  std::vector<int> tokens = tokenizer.Encode(text);

  GPTConfig config;
  config.vocab_size = tokenizer.VocabSize();
  config.block_size = 32;
  config.n_layer = 2;
  config.n_head = 2;
  config.n_embd = 32;

  GPTModel model(config);
  std::cout << "🧠 Modelo GPT C++ Creado exitosamente.\n" << std::flush;

  std::vector<Tensor*> params = model.GetParameters();
  std::vector<Tensor*> grads = model.GetGradients();
  AdamW optimizer(params, grads, 0.001f);
  CrossEntropyLoss criterion;

  int max_iters = 50;
  int batch_size = 4;
  int block_size = config.block_size;

  std::cout << "🏋️ Entrenando durante " << max_iters << " iteraciones en C++...\n" << std::flush;
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int iter = 1; iter <= max_iters; ++iter) {
    optimizer.ZeroGrad();

    Tensor X({batch_size, block_size});
    Tensor Y({batch_size * block_size});

    for (int b = 0; b < batch_size; ++b) {
      int start_idx = (iter * 17 + b * 13) % (tokens.size() - block_size - 1);
      for (int t = 0; t < block_size; ++t) {
        X[b * block_size + t] = static_cast<float>(tokens[start_idx + t]);
        Y[b * block_size + t] = static_cast<float>(tokens[start_idx + t + 1]);
      }
    }

    Tensor logits = model.Forward(X);
    Tensor logits_2d({batch_size * block_size, config.vocab_size});
    std::memcpy(logits_2d.Data(), logits.Data(), logits.TotalSize() * sizeof(float));

    float loss = criterion.Forward(logits_2d, Y);
    optimizer.Step();

    if (iter % 10 == 0 || iter == max_iters) {
      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(current_time - start_time).count();
      std::cout << "Step " << iter << "/" << max_iters << " | Loss LLM: " << loss << " | Tiempo: " << elapsed << "s\n" << std::flush;
    }
  }

  std::cout << "✅ ¡Entrenamiento del LLM en C++ completado exitosamente!\n" << std::flush;
  return 0;
}
