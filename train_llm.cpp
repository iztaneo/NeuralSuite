// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file train_llm.cpp
 * @brief Transformer LLM Training Script matching PyTorch architecture parameters.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "gpt.h"
#include "losses.h"
#include "optimizers.h"
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

  static std::mt19937 rng(42);
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng);
}

void ClipGradients(const std::vector<Tensor*>& grads, float max_norm = 1.0f) {
  float total_norm_sq = 0.0f;
  for (auto g : grads) {
    size_t sz = g->TotalSize();
    for (size_t i = 0; i < sz; ++i) {
      float val = g->operator[](i);
      total_norm_sq += val * val;
    }
  }
  float total_norm = std::sqrt(total_norm_sq);
  if (total_norm > max_norm && total_norm > 0.0f) {
    float scale = max_norm / total_norm;
    for (auto g : grads) {
      size_t sz = g->TotalSize();
      for (size_t i = 0; i < sz; ++i) {
        (*g)[i] *= scale;
      }
    }
  }
}

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

  // Hiperparámetros óptimos para capacidad y velocidad (120,000 parámetros)
  GPTConfig config;
  config.vocab_size = tokenizer.VocabSize();
  config.block_size = 64;
  config.n_layer = 4;
  config.n_head = 4;
  config.n_embd = 64;

  GPTModel model(config);
  std::cout << "🧠 Modelo GPT C++ Creado exitosamente.\n" << std::flush;

  std::vector<Tensor*> params = model.GetParameters();
  std::vector<Tensor*> grads = model.GetGradients();

  float base_lr = 0.003f;
  float min_lr = 0.0001f;
  AdamW optimizer(params, grads, base_lr);
  CrossEntropyLoss criterion;

  int max_iters = 1000;
  int batch_size = 16;
  int block_size = config.block_size;


  std::cout << "🏋️ Entrenando durante " << max_iters << " iteraciones en C++...\n" << std::flush;
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int iter = 1; iter <= max_iters; ++iter) {
    optimizer.ZeroGrad();

    // Cosine decay learning rate
    float decay_ratio = static_cast<float>(iter) / static_cast<float>(max_iters);
    float coeff = 0.5f * (1.0f + std::cos(3.14159265f * decay_ratio));
    float lr = min_lr + coeff * (base_lr - min_lr);
    optimizer.SetLearningRate(lr);

    Tensor X({batch_size, block_size});
    Tensor Y({batch_size * block_size});

    static std::mt19937 data_rng(1337);
    std::uniform_int_distribution<int> data_dist(0, static_cast<int>(tokens.size() - block_size - 1));

    for (int b = 0; b < batch_size; ++b) {
      int start_idx = data_dist(data_rng);
      for (int t = 0; t < block_size; ++t) {
        X[b * block_size + t] = static_cast<float>(tokens[start_idx + t]);
        Y[b * block_size + t] = static_cast<float>(tokens[start_idx + t + 1]);
      }
    }


    Tensor logits = model.Forward(X);
    Tensor logits_2d({batch_size * block_size, config.vocab_size});
    std::memcpy(logits_2d.Data(), logits.Data(), logits.TotalSize() * sizeof(float));

    float loss = criterion.Forward(logits_2d, Y);

    Tensor dlogits_2d = criterion.Backward();
    Tensor dlogits({batch_size, block_size, config.vocab_size});
    std::memcpy(dlogits.Data(), dlogits_2d.Data(), dlogits_2d.TotalSize() * sizeof(float));

    model.Backward(dlogits);
    ClipGradients(grads, 1.0f);
    optimizer.Step();

    if (iter % 25 == 0 || iter == max_iters) {

      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(current_time - start_time).count();
      std::cout << "Step " << iter << "/" << max_iters << " | Loss LLM: " << loss << " | LR: " << lr << " | Tiempo: " << elapsed << "s\n" << std::flush;
    }
  }

  model.SaveWeights("model_cpp.bin");
  std::cout << "💾 Modelo guardado exitosamente en 'model_cpp.bin'.\n" << std::flush;
  std::cout << "✅ ¡Entrenamiento del LLM en C++ completado exitosamente!\n\n" << std::flush;

  std::cout << "============================================================\n" << std::flush;
  std::cout << "🤖 GENERACIÓN DE TEXTO EN C++ DESPUÉS DEL ENTRENAMIENTO\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  std::string prompt = "First Citizen:\n";
  std::vector<int> gen_tokens = tokenizer.Encode(prompt);
  int max_new_tokens = 150;
  int newline_token = tokenizer.Encode("\n")[0];

  for (int step = 0; step < max_new_tokens; ++step) {
    int seq_len = static_cast<int>(gen_tokens.size());
    int start_idx = (seq_len > config.block_size) ? (seq_len - config.block_size) : 0;
    int curr_len = seq_len - start_idx;

    Tensor idx({1, curr_len});
    for (int i = 0; i < curr_len; ++i) {
      idx[i] = static_cast<float>(gen_tokens[start_idx + i]);
    }

    Tensor logits = model.Forward(idx);
    int last_offset = (curr_len - 1) * config.vocab_size;

    int prev_token = gen_tokens.empty() ? -1 : gen_tokens.back();
    int sampled_token = SampleToken(&logits[last_offset], config.vocab_size, 0.7f, prev_token, newline_token);
    gen_tokens.push_back(sampled_token);
  }

  std::string generated = tokenizer.Decode(gen_tokens);
  std::cout << generated << "\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
