// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file imgdump.cpp
 * @brief Decodifica una imagen y vuelca sus pixeles en crudo.
 *
 * No sirve para nada por si solo: existe para que compare_pillow.py pueda
 * contrastar byte a byte lo que decodifica NeuralSuite contra lo que decodifica
 * Pillow. Escribe una linea con "ancho alto canales" y a continuacion los
 * bytes, sin cabecera ni relleno.
 */

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "image.h"
using namespace neuralsuite::image;
int main(int argc, char** argv) {
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Bitmap bmp; std::string err;
  if (!Decode(data.data(), data.size(), &bmp, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::ofstream out(argv[2], std::ios::binary);
  out << bmp.width << " " << bmp.height << " " << bmp.channels << "\n";
  out.write(reinterpret_cast<const char*>(bmp.pixels.data()), bmp.pixels.size());
  return 0;
}
