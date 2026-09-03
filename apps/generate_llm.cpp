// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file generate_llm.cpp
 * @brief Autoregressive Text Generation Demo with CLI flags matching PyTorch options.
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "artifacts.h"
#include "gpt.h"
#include "tensor.h"
#include "tokenizer.h"

using namespace neuralsuite;

struct GenerateArgs {
  std::string prompt = "First Citizen:\n";
  // Los artefactos de entrenamiento viven bajo release/ (ver include/artifacts.h).
  std::string model_path = ReleasePath("model_cpp.bin");
  std::string vocab_path = ReleasePath("vocab_cpp.txt");
  int max_new_tokens = 200;
  float temperature = 0.7f;
  int block_size = 64;
  int n_layer = 4;
  int n_head = 4;
  int n_embd = 128;

  bool use_cache = true;
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
            << "  --n_embd <int>            Dimensión del embedding (default: 128)\n"
            << "  --no_cache                Desactivar aceleración por KV-Cache\n"
            << "  --model_path <path>       Ruta al archivo binario del modelo (default: release/model_cpp.bin)\n"
            << "  --vocab_path <path>       Ruta al archivo de vocabulario (default: release/vocab_cpp.txt)\n"
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
    } else if (arg == "--no_cache") {
      args.use_cache = false;
    } else if (arg == "--model_path" && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if (arg == "--vocab_path" && i + 1 < argc) {
      args.vocab_path = argv[++i];
    } else if (i == 1 && arg[0] != '-') {
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
  std::cout << "⚡ Aceleración por KV-Cache: " << (args.use_cache ? "ACTIVADA (Inferencia O(1))" : "DESACTIVADA") << "\n" << std::flush;
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

  if (args.use_cache) {
    // La caché guarda K y V de cada posición ya vista, así que sólo hace falta
    // alimentar el token nuevo. Dos cosas hay que respetar, y ninguna es obvia:
    //
    //  - No reinyectar. Cada token entra en la caché UNA vez. Alimentar otra vez
    //    el último token del prompt lo duplicaba dentro de la caché, y los
    //    logits salían distintos de los de la ruta sin caché: medido, 0.056 de
    //    diferencia, frente a 0.000000 sin reinyectar. No se veía porque
    //    comparar la secuencia generada no basta —el argmax absorbe una
    //    diferencia pequeña— y hay que comparar los logits.
    //
    //  - Deslizar la ventana. `wpe_` sólo tiene `block_size` posiciones, así que
    //    pasado ese punto la posición se sale de la tabla y antes reventaba con
    //    `Embedding: token N fuera del rango`. La caché se reconstruye con los
    //    últimos `block_size` tokens, que es justo lo que hace la ruta sin caché
    //    al recortar el contexto.
    //
    // El precio de deslizar hay que decirlo: **pasada la ventana, la caché deja
    // de acelerar**. Medido con `block_size` 32: 0.13 ms/token dentro de la
    // ventana y 1.29 ms/token al cruzarla, con 73 reconstrucciones en 100
    // tokens, o sea casi una por paso.
    //
    // No es un descuido del código sino una consecuencia de usar posiciones
    // **aprendidas y absolutas**: al deslizar, cada token cambia de posición, y
    // las K y V guardadas llevan la posición dentro, así que dejan de valer.
    // Reconstruir menos a menudo —tirando media ventana en vez de una posición—
    // sería más rápido, pero acortaría el contexto y las dos rutas dejarían de
    // dar el mismo resultado, que es justo lo que fija el test 38.
    //
    // La solución de fondo es RoPE, donde la posición es relativa y deslizar no
    // invalida nada. Es un argumento concreto a favor de la Fase 15 del roadmap.
    auto sembrar_cache = [&](int desde, int hasta) {
      Tensor ultimo;
      model.ClearKVCache();
      for (int i = desde; i < hasta; ++i) {
        ultimo = model.ForwardWithKVCache(tokens[i], i - desde);
      }
      return ultimo;
    };

    int inicio = std::max(0, static_cast<int>(tokens.size()) - config.block_size);
    Tensor logits = sembrar_cache(inicio, static_cast<int>(tokens.size()));

    for (int step = 0; step < args.max_new_tokens; ++step) {
      const int prev_token = tokens.empty() ? -1 : tokens.back();
      const int sampled_token = SampleToken(logits.Data(), config.vocab_size,
                                            args.temperature, prev_token, newline_token);
      tokens.push_back(sampled_token);

      const int seq_len = static_cast<int>(tokens.size());
      if (seq_len - inicio > config.block_size) {
        inicio = seq_len - config.block_size;
        logits = sembrar_cache(inicio, seq_len);
      } else {
        logits = model.ForwardWithKVCache(sampled_token, seq_len - 1 - inicio);
      }
    }
  } else {
    // Modo tradicional sin caché
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
  }


  std::string generated = tokenizer.Decode(tokens);
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "RESULTADO GENERADO POR EL LLM EN C++:\n" << std::flush;
  std::cout << generated << "\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;

  return 0;
}
