// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file benchmark.cpp
 * @brief Mide el rendimiento de las operaciones que dominan el tiempo.
 *
 * Las pruebas de `test_suite` responden si el calculo es correcto. Esto
 * responde si es rapido, que es una pregunta distinta y con otro criterio de
 * exito: aqui no hay nada que "pase" o "falle", solo numeros que comparar entre
 * versiones.
 *
 * Existe porque las mediciones que guiaron la optimizacion se hicieron a mano y
 * no quedaban en el repositorio: nadie podia reproducirlas ni notar una
 * regresion. Las cifras que cita docs/ROADMAP.md salen de aqui.
 *
 * Uso:
 *   ./benchmark            todas las mediciones
 *   ./benchmark --gemm     solo la multiplicacion de matrices
 *   ./benchmark --threads  solo el escalado con hilos
 *   ./benchmark --train    solo el paso de entrenamiento
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "neuralsuite.h"
#include "parallel.h"

using namespace neuralsuite;

namespace {

/** @brief Mediana de varias repeticiones, en milisegundos. */
template <typename Fn>
double MedianMs(Fn&& fn, int reps) {
  std::vector<double> samples;
  samples.reserve(reps);
  fn();  // calentamiento: la primera pasada paga las reservas y los fallos de cache
  for (int i = 0; i < reps; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void BenchGemm() {
  std::printf("\n=== MatMul (%d hilos) ===\n", parallel::ThreadCount());
  std::printf("  %-22s %10s %12s\n", "forma", "ms", "GFLOP/s");

  const struct { int m, k, n; const char* note; } cases[] = {
      {512, 128, 384, "c_attn del GPT"},
      {512, 128, 512, "mlp_fc"},
      {512, 512, 128, "mlp_proj"},
      {512, 512, 512, ""},
      {1024, 512, 512, ""},
  };

  for (const auto& c : cases) {
    Tensor A({c.m, c.k}), B({c.k, c.n}), C;
    A.RandomNormal(0.0f, 1.0f);
    B.RandomNormal(0.0f, 1.0f);
    const double ms = MedianMs([&] { MatMul(A, B, C); }, 20);
    char shape[32];
    std::snprintf(shape, sizeof(shape), "%dx%dx%d", c.m, c.k, c.n);
    std::printf("  %-22s %10.3f %12.1f  %s\n", shape, ms,
                2.0 * c.m * c.k * c.n / (ms * 1e6), c.note);
  }
}

void BenchThreadScaling() {
  std::printf("\n=== Escalado con hilos (MatMul 1024x512x512) ===\n");
  std::printf("  %-10s %10s %10s\n", "hilos", "ms", "aceleracion");

  Tensor A({1024, 512}), B({512, 512}), C;
  A.RandomNormal(0.0f, 1.0f);
  B.RandomNormal(0.0f, 1.0f);

  const int original = parallel::ThreadCount();
  double baseline = 0.0;
  for (int threads : {1, 2, 4, original}) {
    if (threads > original) continue;
    parallel::ThreadCount() = threads;
    const double ms = MedianMs([&] { MatMul(A, B, C); }, 15);
    if (threads == 1) baseline = ms;
    std::printf("  %-10d %10.3f %9.2fx\n", threads, ms, baseline / ms);
  }
  parallel::ThreadCount() = original;

  // El reparto no altera el resultado: cada hilo escribe filas disjuntas, de
  // modo que no hay reduccion que cambie el orden de las sumas.
  parallel::ThreadCount() = 1;
  Tensor serial;
  MatMul(A, B, serial);
  parallel::ThreadCount() = original;
  Tensor threaded;
  MatMul(A, B, threaded);

  double max_diff = 0.0;
  for (size_t i = 0; i < serial.TotalSize(); ++i) {
    max_diff = std::max(max_diff, static_cast<double>(std::abs(serial[i] - threaded[i])));
  }
  std::printf("  diferencia frente al calculo en un hilo: %.1e %s\n", max_diff,
              max_diff == 0.0 ? "(identico bit a bit)" : "(ATENCION: no es identico)");
}

void BenchTrainingStep() {
  std::printf("\n=== Paso de entrenamiento del GPT ===\n");

  const struct { int layers, embd, batch, seq; } configs[] = {
      {2, 64, 4, 32},
      {4, 128, 8, 64},
  };

  for (const auto& cfg_case : configs) {
    GPTConfig cfg;
    cfg.vocab_size = 65;
    cfg.block_size = cfg_case.seq;
    cfg.n_layer = cfg_case.layers;
    cfg.n_head = 4;
    cfg.n_embd = cfg_case.embd;

    GPTModel model(cfg);
    const int B = cfg_case.batch, T = cfg_case.seq;

    Tensor idx({B, T});
    for (int i = 0; i < B * T; ++i) idx[i] = static_cast<float>(i % cfg.vocab_size);
    Tensor targets({B * T});
    for (int i = 0; i < B * T; ++i) targets[i] = static_cast<float>((i * 7) % cfg.vocab_size);

    CrossEntropyLoss criterion;
    const double ms = MedianMs([&] {
      Tensor logits = model.Forward(idx);
      Tensor logits_2d = logits;
      logits_2d.Reshape({B * T, cfg.vocab_size});
      criterion.Forward(logits_2d, targets);
      Tensor dlogits = criterion.Backward();
      dlogits.Reshape({B, T, cfg.vocab_size});
      model.Backward(dlogits);
    }, 10);

    const double tokens_per_s = (B * T) / (ms / 1000.0);
    std::printf("  %d capas, n_embd=%-4d lote %dx%-3d  %8.2f ms  %10.0f tokens/s\n",
                cfg.n_layer, cfg.n_embd, B, T, ms, tokens_per_s);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool all = argc == 1;
  bool gemm = all, threads = all, train = all;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--gemm") gemm = true;
    else if (arg == "--threads") threads = true;
    else if (arg == "--train") train = true;
    else if (arg == "--help") {
      std::printf("Uso: benchmark [--gemm] [--threads] [--train]\n");
      return 0;
    }
  }

  std::printf("============================================================\n");
  std::printf("Benchmarks de NeuralSuite (%d hilos disponibles)\n", parallel::ThreadCount());
  std::printf("============================================================\n");
  std::printf("Las cifras dependen de la maquina; lo que importa es compararlas\n");
  std::printf("entre versiones del mismo repositorio, no en terminos absolutos.\n");

  if (gemm) BenchGemm();
  if (threads) BenchThreadScaling();
  if (train) BenchTrainingStep();

  std::printf("\n");
  return 0;
}
