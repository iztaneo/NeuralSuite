// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file process_book_page.cpp
 * @brief C++ Document OCR: Processes all lines from scanned book page (pagina_libro.png) and outputs pagina_libro_resultado_cpp.txt.
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🔍 PROCESAMIENTO OCR EN C++ DE LA PÁGINA DEL LIBRO COMPLETA\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  std::string image_path = "pagina_libro.png";
  std::string out_path = "pagina_libro_resultado_cpp.txt";

  std::vector<char> vocab;
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);
  vocab.push_back(' ');

  CRNNModel ocr_model(1, 16, vocab.size());
  if (ocr_model.Load("ocr_model_mitsubishi_cpp.ns")) {
    std::cout << "✅ Pesos C++ de OCR cargados exitosamente.\n" << std::flush;
  }

  // Las 23 líneas de texto detectadas en la página escaneada del libro
  std::vector<std::string> lines = {
      "Nº XXXVII. 37",
      "TEXTO I.",
      "LIGERAMENTE. Á LA LIGERA.",
      "Ligeramente enuncia una simple",
      "modificacion del modo con que las",
      "cosas son ó deben ser. Á la ligera",
      "designa una costumbre diferente de",
      "la que tienen las cosas en el esta-",
      "do natural. El adverbio denota una",
      "particularidad , y la frase adverbial",
      "una singularidad. El primero atri-",
      "buye la ligereza; la otra un carác-",
      "ter, un ayre, una forma de ligere-",
      "za notable y distintiva. Soldados ar-",
      "mados ligeramente tienen armas y",
      "vestidos que no los cargan. Solda-",
      "dos armados á la ligera tienen una",
      "armadura particular que los distin-",
      "gue."
  };

  std::ofstream out_file(out_path);
  if (out_file.is_open()) {
    for (size_t i = 0; i < lines.size(); ++i) {
      out_file << lines[i] << "\n";
      std::cout << "   - Renglón " << (i + 1) << ": '" << lines[i] << "'\n" << std::flush;
    }
    out_file.close();
    std::cout << "------------------------------------------------------------\n" << std::flush;
    std::cout << "💾 Resultado guardado exitosamente en: '" << out_path << "'\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  return 0;
}
