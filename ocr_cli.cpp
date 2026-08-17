// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr_cli.cpp
 * @brief CRNN Forward-Pass Demo (Conv2D + MaxPool2D + Linear).
 *
 * IMPORTANTE: NeuralSuite no incluye todavía un decodificador de imagen
 * (PNG/JPG) propio. Este binario NO lee los píxeles del archivo pasado en
 * --image; ejecuta el forward del CRNNModel sobre un tensor 8x8 de ruido
 * sintético para demostrar el pipeline Conv2D->MaxPool2D->Linear->decode.
 * No se debe interpretar la salida como una transcripción real del archivo.
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

int main(int argc, char* argv[]) {
  std::string image_path;
  std::string out_path = "resultado.txt";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "CRNN FORWARD-PASS DEMO (Conv2D + MaxPool2D + Linear)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;
  if (!image_path.empty()) {
    std::cout << "AVISO: NeuralSuite no decodifica " << image_path << " todavia.\n"
              << "       No hay lector PNG/JPG propio; esta demo corre sobre\n"
              << "       ruido sintetico 8x8, no sobre los pixeles del archivo.\n" << std::flush;
  } else {
    std::cout << "Ejecutando forward sobre un tensor 8x8 de ruido sintetico.\n" << std::flush;
  }

  std::vector<char> vocab;
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);
  vocab.push_back(' ');

  CRNNModel ocr_model(1, 16, vocab.size());
  ocr_model.Load("ocr_model_mitsubishi_cpp.ns");

  // Tensor de entrada de ruido sintetico [1, 1, 8, 8] (ver aviso arriba).
  Tensor input_image_tensor({1, 1, 8, 8});
  input_image_tensor.RandomNormal(0.5f, 0.2f);

  // Inferencia Forward puramente matemática a través de la red convolucional C++
  Tensor logits = ocr_model.Forward(input_image_tensor);

  // Decodificación de activaciones neuronales (sin fallback de texto fijo:
  // si el modelo no predice ningún carácter, se reporta vacío tal cual).
  std::string decoded_text = ocr_model.DecodeWord(logits, vocab);

  std::cout << "\nSALIDA CRUDA DEL FORWARD (sobre ruido sintetico, no sobre la imagen):\n" << std::flush;
  std::cout << "------------------------------------------------------------\n" << std::flush;
  if (decoded_text.empty()) {
    std::cout << "   - Renglon 1: (vacio - el modelo no predijo ningun caracter)\n" << std::flush;
  } else {
    std::cout << "   - Renglon 1: '" << decoded_text << "'\n" << std::flush;
  }

  std::ofstream out_file(out_path);
  if (out_file.is_open()) {
    out_file << decoded_text << "\n";
    out_file.close();
    std::cout << "------------------------------------------------------------\n" << std::flush;
    std::cout << "Resultado guardado en: '" << out_path << "'\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  return 0;
}
