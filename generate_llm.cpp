// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file generate_llm.cpp
 * @brief Autoregressive Text Generation Demo with CLI flags matching PyTorch options.
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

struct GenerateArgs {
  std::string prompt = "First Citizen:\n";
  std::string model_path = "model_cpp.bin";
  std::string vocab_path = "vocab_cpp.txt";
  int max_new_tokens = 200;
  float temperature = 0.7f;
  int block_size = 64;
  int n_layer = 4;
  int n_head = 4;
  int n_embd = 128;

};

void PrintGenerateUsage(const char* prog_name) {
  std::cout << "Uso: " << prog_name << " [OPCIONES] [\"Prompt Opcional\"]\n\n"
            << "Opciones Disponibles (equivalentes a PyTorch generate.py):\n"
            << "  --prompt <string>         Texto inicial para la generación (default: \"First Citizen:\\n\")\n"
            << "  --max_new_tokens <int>    Número de caracteres a generar (default: 200)\n"
            << "  --temperature <float>    Temperatura de muestreo (default: 0.7)\n"
            << "  --block_size <int>        Longitud del contexto (default: 64)\n"
            << "  --n_layer <int>           Número de capas del modelo (default: 4)\n"
            << "  --n_head <int>            Número de cabezas de atención (default: 4)\n"
            << "  --n_embd <int>            Dimensión del embedding (default: 64)\n"
            << "  --model_path <path>       Ruta al archivo binario del modelo (default: model_cpp.bin)\n"
            << "  --vocab_path <path>       Ruta al archivo de vocabulario (default: vocab_cpp.txt)\n"
            << "  --help                    Muestra este mensaje de ayuda\n";
}

GenerateArgs ParseGenerateArgs(int argc, char** argv) {
  GenerateArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintGenerateUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--prompt" && i + 1 < argc) {
      args.prompt = argv[++i];
    } else if (arg == "--max_new_tokens" && i + 1 < argc) {
      args.max_new_tokens = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      args.temperature = std::stof(argv[++i]);
    } else if (arg == "--block_size" && i + 1 < argc) {
      args.block_size = std::stoi(argv[++i]);
    } else if (arg == "--n_layer" && i + 1 < argc) {
      args.n_layer = std::stoi(argv[++i]);
    } else if (arg == "--n_head" && i + 1 < argc) {
      args.n_head = std::stoi(argv[++i]);
    } else if (arg == "--n_embd" && i + 1 < argc) {
      args.n_embd = std::stoi(argv[++i]);
    } else if (arg == "--model_path" && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if (arg == "--vocab_path" && i + 1 < argc) {
      args.vocab_path = argv[++i];
    } else if (i == 1 && arg[0] != '-') {
      // Positional argument fallback for prompt string
      args.prompt = arg;
    }
  }
  return args;
}

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
  GenerateArgs args = ParseGenerateArgs(argc, argv);

  std::cout << "============================================================\n" << std::flush;
  std::cout << "🤖 GENERACIÓN DE TEXTO AUTORREGRESIVA C++ (NeuralSuite CLI)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  CharTokenizer tokenizer;
  if (!tokenizer.Load(args.vocab_path)) {
    std::cout << "⚠️ No se encontró '" << args.vocab_path << "'. Usando vocabulario por defecto...\n" << std::flush;
    tokenizer.BuildVocab("First Citizen:\nBefore we proceed any further, hear me speak.");
  } else {
    std::cout << "🔤 Vocabulario cargado desde '" << args.vocab_path << "': " << tokenizer.VocabSize() << " caracteres.\n" << std::flush;
  }

  GPTConfig config;
  config.vocab_size = tokenizer.VocabSize();
  config.block_size = args.block_size;
  config.n_layer = args.n_layer;
  config.n_head = args.n_head;
  config.n_embd = args.n_embd;

  GPTModel model(config);
  if (model.LoadWeights(args.model_path)) {
    std::cout << "✅ Pesos del modelo C++ cargados desde '" << args.model_path << "'.\n" << std::flush;
  } else {
    std::cout << "⚠️ No se encontró '" << args.model_path << "'. Usando pesos aleatorios...\n" << std::flush;
  }

  std::cout << "\nPrompt de entrada: '" << args.prompt << "'\n" << std::flush;

  std::vector<int> tokens = tokenizer.Encode(args.prompt);
  int newline_token = tokenizer.Encode("\n")[0];

  for (int step = 0; step < args.max_new_tokens; ++step) {
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
    int sampled_token = SampleToken(&logits[last_offset], config.vocab_size, args.temperature, prev_token, newline_token);
    tokens.push_back(sampled_token);
  }

  std::string generated = tokenizer.Decode(tokens);
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "RESULTADO GENERADO POR EL LLM EN C++:\n" << std::flush;
  std::cout << generated << "\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;

  return 0;
}
