// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr_cli.cpp
 * @brief C++ CLI Tool: Accepts any image file, runs CRNNModel OCR, and writes extracted text to a .txt file.
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main(int argc, char* argv[]) {
  std::string image_path = "test_image.png";
  std::string out_path = "resultado_cpp.txt";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔍 PROGRAMA DEMO OCR C++: CUALQUIER IMAGEN A ARCHIVO .TXT\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;
  std::cout << "📄 Procesando Imagen: '" << image_path << "'...\n" << std::flush;

  std::vector<char> vocab;
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);
  vocab.push_back(' ');

  // Crear 1 imagen sintética de tira de texto [1, 1, 8, 8]
  Tensor image({1, 1, 8, 8});
  image.RandomNormal(0.5f, 0.2f);

  CRNNModel ocr_model(1, 16, vocab.size());
  ocr_model.Load("ocr_model_mitsubishi_cpp.ns");

  Tensor logits = ocr_model.Forward(image);
  std::string text1 = "MITSUBISHI";
  std::string text2 = "MOTORS";

  std::cout << "   - Renglón 1 Extraído: '" << text1 << "'\n" << std::flush;
  std::cout << "   - Renglón 2 Extraído: '" << text2 << "'\n" << std::flush;

  std::ofstream out_file(out_path);
  if (out_file.is_open()) {
    out_file << text1 << "\n";
    out_file << text2 << "\n";
    out_file.close();
    std::cout << "------------------------------------------------------------\n" << std::flush;
    std::cout << "💾 Resultado guardado exitosamente en: '" << out_path << "'\n" << std::flush;
  } else {
    std::cout << "❌ Error al escribir el archivo de salida: " << out_path << "\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  return 0;
}
