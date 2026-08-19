// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr_cli.cpp
 * @brief Forward del CRNN sobre una linea sintetica (Conv x3 -> BiLSTM -> Linear).
 *
 * IMPORTANTE: NeuralSuite todavia no decodifica imagenes. Este binario NO lee
 * los pixeles del archivo que se le pase en --image; genera una linea sintetica
 * con SynthTextGenerator y ejecuta el forward sobre ella. La salida no es una
 * transcripcion del archivo, y el aviso se imprime siempre para que no pueda
 * confundirse con una.
 */

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main(int argc, char* argv[]) {
  std::string image_path;
  std::string out_path = "resultado.txt";
  int word_len = 10;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    if (arg == "--len" && i + 1 < argc) word_len = std::stoi(argv[++i]);
    if (arg == "--help") {
      std::cout << "Uso: ocr_cli [--image ruta] [--out ruta] [--len n]\n";
      return 0;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "CRNN: Conv x3 -> BiLSTM -> Linear -> decode\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;
  if (!image_path.empty()) {
    std::cout << "AVISO: NeuralSuite no decodifica " << image_path << " todavia.\n"
              << "       No hay lector PNG/JPG propio, asi que este forward corre\n"
              << "       sobre una linea sintetica, no sobre los pixeles del archivo.\n"
              << std::flush;
  } else {
    std::cout << "Forward sobre una linea sintetica de " << word_len << " caracteres.\n"
              << std::flush;
  }

  const std::vector<char> vocab = CRNNModel::DefaultVocab();
  CRNNModel ocr_model(1, 16, static_cast<int>(vocab.size()));

  const std::string weights_path = ReleasePath("ocr_model_mitsubishi_cpp.ns");
  if (!ocr_model.Load(weights_path)) {
    std::cout << "AVISO: no se encontraron pesos en '" << weights_path << "';\n"
              << "       el modelo corre con inicializacion aleatoria.\n" << std::flush;
  }

  Tensor image, targets;
  SynthTextGenerator::Generate(image, targets, /*batch=*/1, word_len,
                               static_cast<int>(vocab.size()));

  const Tensor logits = ocr_model.Forward(image);
  const std::string decoded_text = ocr_model.DecodeWord(logits, vocab);

  std::cout << "\nSALIDA DEL FORWARD (sobre la linea sintetica, no sobre la imagen):\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;
  std::cout << "   - entrada  : " << CRNNModel::kInputHeight << "x" << image.Shape()[3]
            << " px  ->  " << CRNNModel::TimestepsFor(image.Shape()[3]) << " pasos\n" << std::flush;
  if (decoded_text.empty()) {
    std::cout << "   - renglon 1: (vacio - el modelo no predijo ningun caracter)\n" << std::flush;
  } else {
    std::cout << "   - renglon 1: '" << decoded_text << "'\n" << std::flush;
  }

  std::ofstream out_file(out_path);
  if (out_file.is_open()) {
    out_file << decoded_text << "\n";
    std::cout << "------------------------------------------------------------\n" << std::flush;
    std::cout << "Resultado guardado en: '" << out_path << "'\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  return 0;
}
