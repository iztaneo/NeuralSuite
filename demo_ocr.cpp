// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_ocr.cpp
 * @brief Entrena el CRNN a leer una linea de texto sintetica y la decodifica.
 *
 * Los datos son sinteticos —cada caracter se dibuja como un patron binario
 * derivado de su indice, no con una tipografia real— pero el problema si lo es:
 * la red recibe una imagen de una linea entera y tiene que devolver la secuencia
 * de caracteres, no clasificar una imagen suelta. Que la palabra decodificada
 * coincida con la esperada es una comprobacion de extremo a extremo.
 */

#include <iostream>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔍 Demostración 9: OCR de una línea con CRNN\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  const std::vector<char> vocab = {'A', 'B', 'C', 'D', 'E', 'F'};
  const int kBatch = 4, kWordLen = 6, kEpochs = 120;

  ManualSeed(7);
  Tensor images, targets;
  SynthTextGenerator::Generate(images, targets, kBatch, kWordLen,
                               static_cast<int>(vocab.size()), /*seed=*/2024);

  std::cout << "Lote: " << kBatch << " imágenes de "
            << CRNNModel::kInputHeight << "x" << images.Shape()[3]
            << " px, " << kWordLen << " caracteres cada una.\n" << std::flush;
  std::cout << "La red devuelve " << CRNNModel::TimestepsFor(images.Shape()[3])
            << " predicciones por imagen: una por columna que sobrevive al pooling.\n\n"
            << std::flush;

  CRNNModel ocr_model(1 /*canales*/, 32 /*ocultas*/, static_cast<int>(vocab.size()));

  CrossEntropyLoss criterion;
  AdamW optimizer(ocr_model.Parameters(), 0.01f);

  // La perdida compara paso a paso: [B*T, clases] contra [B*T] objetivos.
  Tensor flat_targets = targets;
  flat_targets.Reshape({kBatch * kWordLen});

  std::cout << "🏋️ Entrenando " << kEpochs << " épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= kEpochs; ++epoch) {
    optimizer.ZeroGrad();

    Tensor logits = ocr_model.Forward(images);
    Tensor logits_2d = logits;
    logits_2d.Reshape({kBatch * kWordLen, static_cast<int>(vocab.size())});

    const float loss = criterion.Forward(logits_2d, flat_targets);

    Tensor dlogits = criterion.Backward();
    dlogits.Reshape(logits.Shape());
    ocr_model.Backward(dlogits);

    optimizer.Step();

    if (epoch % 20 == 0 || epoch == 1) {
      std::cout << "  época " << epoch << "/" << kEpochs << " | pérdida: " << loss << "\n"
                << std::flush;
    }
  }

  const std::string ocr_path = ReleasePath("ocr_model.ns");
  if (ocr_model.Save(ocr_path)) {
    std::cout << "\n💾 Pesos guardados en '" << ocr_path << "'.\n" << std::flush;
  } else {
    std::cerr << "\nERROR: no se pudieron guardar los pesos en '" << ocr_path << "'.\n" << std::flush;
  }

  std::cout << "\n🎯 Transcripción de las " << kBatch << " líneas:\n" << std::flush;
  const Tensor final_logits = ocr_model.Forward(images);
  const std::vector<std::string> decoded = ocr_model.DecodeBatch(final_logits, vocab);

  int exactas = 0;
  for (int b = 0; b < kBatch; ++b) {
    std::string esperado;
    for (int t = 0; t < kWordLen; ++t) {
      esperado += vocab[static_cast<int>(targets[static_cast<size_t>(b) * kWordLen + t])];
    }
    // El decodificado colapsa repeticiones consecutivas, asi que una palabra con
    // dos letras iguales seguidas no puede reproducirse tal cual: se compara
    // contra la esperada ya colapsada.
    std::string esperado_colapsado;
    for (size_t i = 0; i < esperado.size(); ++i) {
      if (i == 0 || esperado[i] != esperado[i - 1]) esperado_colapsado += esperado[i];
    }
    const bool ok = decoded[b] == esperado_colapsado;
    if (ok) ++exactas;
    std::cout << "   " << (ok ? "✅" : "❌") << " leído '" << decoded[b]
              << "'  esperado '" << esperado_colapsado << "'"
              << (esperado_colapsado == esperado ? "" : "  (original '" + esperado + "')")
              << "\n" << std::flush;
  }

  std::cout << "\n" << exactas << "/" << kBatch << " líneas transcritas exactamente.\n"
            << std::flush;
  std::cout << "============================================================\n" << std::flush;
  std::cout << "Nota: los glifos son patrones sintéticos, no una tipografía\n" << std::flush;
  std::cout << "real, y no hay lector de PNG/JPG todavía. Lo que se demuestra\n" << std::flush;
  std::cout << "es la arquitectura completa: Conv×3 → BiLSTM → Linear → decode.\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
