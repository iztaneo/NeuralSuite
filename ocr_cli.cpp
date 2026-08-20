// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr_cli.cpp
 * @brief Lee una imagen del disco y la pasa por el CRNN.
 *
 * Hasta ahora este binario no abria el archivo que se le daba en --image:
 * anunciaba que no podia y corria sobre datos sinteticos. Ya no. Con
 * include/image.h decodifica PNG, BMP y Netpbm, lo pasa a gris, lo reescala a
 * los 32 pixeles de alto que espera la red y ejecuta el forward.
 *
 * Queda una limitacion, y conviene no disimularla: el modelo no viene entrenado
 * con tipografias reales. La imagen se lee de verdad, pero la transcripcion
 * saldra sin sentido mientras no haya pesos entrenados sobre texto real. Lo que
 * ya funciona es todo el camino: archivo -> pixeles -> tensor -> red -> texto.
 */

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

namespace {

void PrintUsage() {
  std::cout << "Uso: ocr_cli [--image ruta] [--out ruta] [--len n] [--no-invertir]\n"
            << "\n"
            << "  --image ruta   imagen a transcribir (PNG, JPEG, BMP, PBM/PGM/PPM).\n"
            << "                 Sin este argumento se genera una linea sintetica.\n"
            << "  --out ruta     donde escribir el texto (por defecto resultado.txt).\n"
            << "  --len n        caracteres de la linea sintetica (por defecto 10).\n"
            << "  --pesos ruta   archivo de pesos (por defecto release/ocr_texto.ns).\n"
            << "  --oculto n     tamano oculto del modelo; debe coincidir con los pesos.\n"
            << "  --no-invertir  no invertir el gris. Por defecto se invierte, porque\n"
            << "                 un documento trae tinta oscura sobre papel claro y la\n"
            << "                 red espera trazo claro sobre fondo oscuro.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string image_path;
  std::string out_path = "resultado.txt";
  int word_len = 10;
  bool invert = true;
  int oculto = 64;
  std::string pesos = ReleasePath("ocr_texto.ns");

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--len" && i + 1 < argc) {
      word_len = std::stoi(argv[++i]);
    } else if (arg == "--pesos" && i + 1 < argc) {
      pesos = argv[++i];
    } else if (arg == "--oculto" && i + 1 < argc) {
      oculto = std::stoi(argv[++i]);
    } else if (arg == "--no-invertir") {
      invert = false;
    } else if (arg == "--help") {
      PrintUsage();
      return 0;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "CRNN: Conv x3 -> BiLSTM -> Linear -> decode\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  const std::vector<char> vocab = CRNNModel::DefaultVocab();
  CRNNModel ocr_model(1, oculto, static_cast<int>(vocab.size()));

  const std::string weights_path = pesos;
  const bool trained = ocr_model.Load(weights_path);
  if (!trained) {
    std::cout << "AVISO: no se pudieron cargar pesos de '" << weights_path << "';\n"
              << "       el modelo corre con inicializacion aleatoria y lo que salga\n"
              << "       no significara nada. Entrena con ./train_ocr o pasa --pesos.\n"
              << std::flush;
  }

  Tensor input;
  if (!image_path.empty()) {
    // Se lee el archivo entero para poder informar del formato detectado: el
    // formato lo decide el contenido, no la extension.
    std::ifstream probe(image_path, std::ios::binary);
    if (!probe) {
      std::cerr << "ERROR: no se pudo abrir '" << image_path << "'.\n";
      return 1;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(probe)),
                                     std::istreambuf_iterator<char>());
    image::Bitmap bitmap;
    std::string error;
    if (!image::Decode(bytes.data(), bytes.size(), &bitmap, &error)) {
      std::cerr << "ERROR al leer '" << image_path << "': " << error << "\n";
      return 1;
    }
    std::cout << "Imagen : " << image_path << "\n"
              << "Formato: " << image::FormatName(image::DetectFormat(bytes.data(), bytes.size()))
              << ", " << bitmap.width << "x" << bitmap.height << " px, " << bitmap.channels
              << " canal(es)\n" << std::flush;

    if (!image::LoadAsTensor(image_path, CRNNModel::kInputHeight, CRNNModel::kWidthReduction,
                             invert, &input, &error)) {
      std::cerr << "ERROR: " << error << "\n";
      return 1;
    }
  } else {
    std::cout << "Sin --image: se genera una linea sintetica de " << word_len << " caracteres.\n"
              << std::flush;
    Tensor targets;
    SynthTextGenerator::Generate(input, targets, /*batch=*/1, word_len,
                                 static_cast<int>(vocab.size()));
  }

  const Tensor logits = ocr_model.Forward(input);
  const std::string decoded_text = ocr_model.DecodeWord(logits, vocab);

  std::cout << "\n------------------------------------------------------------\n" << std::flush;
  std::cout << "   entrada al modelo : " << CRNNModel::kInputHeight << "x" << input.Shape()[3]
            << " px  ->  " << CRNNModel::TimestepsFor(input.Shape()[3]) << " pasos\n" << std::flush;
  if (decoded_text.empty()) {
    std::cout << "   renglon 1         : (vacio - el modelo no predijo ningun caracter)\n"
              << std::flush;
  } else {
    std::cout << "   renglon 1         : '" << decoded_text << "'\n" << std::flush;
  }
  std::cout << "------------------------------------------------------------\n" << std::flush;

  if (!trained && !image_path.empty()) {
    std::cout << "\nLa imagen se ha leido y reescalado de verdad, pero sin pesos\n"
              << "entrenados la transcripcion no es una lectura del texto, solo la\n"
              << "salida de una red sin entrenar.\n" << std::flush;
  }

  std::ofstream out_file(out_path);
  if (out_file.is_open()) {
    out_file << decoded_text << "\n";
    std::cout << "\nResultado guardado en: '" << out_path << "'\n" << std::flush;
  }

  std::cout << "============================================================\n" << std::flush;
  return 0;
}
