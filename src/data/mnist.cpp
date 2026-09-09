// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de data/mnist.h.

#include "data/mnist.h"

#include <fstream>

namespace neuralsuite {
namespace data {

namespace {

// Los enteros del formato IDX van en BIG-ENDIAN. Leerlos byte a byte funciona
// en cualquier maquina; un memcpy daria la vuelta al numero en las
// little-endian, que son todas las que soporta el proyecto.
uint32_t LeerU32BE(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

bool LeerArchivo(const std::string& ruta, std::vector<uint8_t>* datos, std::string* error) {
  std::ifstream f(ruta, std::ios::binary);
  if (!f) {
    *error = "no se pudo abrir '" + ruta + "'.";
    return false;
  }
  f.seekg(0, std::ios::end);
  const std::streamoff tam = f.tellg();
  f.seekg(0, std::ios::beg);
  if (tam <= 0) {
    *error = "'" + ruta + "' esta vacio.";
    return false;
  }
  datos->resize(static_cast<size_t>(tam));
  f.read(reinterpret_cast<char*>(datos->data()), tam);
  return true;
}

}  // namespace

bool LeerIdxImagenes(const std::string& ruta, Tensor* salida, int* n, std::string* error,
                     bool normalizar) {
  std::vector<uint8_t> datos;
  if (!LeerArchivo(ruta, &datos, error)) return false;
  // El numero magico se comprueba PRIMERO, aunque la cabecera completa sean 16
  // bytes: solo hacen falta 4 para leerlo, y es el mensaje mas util. Al reves
  // —comprobando el tamano antes— pasar el archivo de etiquetas como si fuera
  // el de imagenes daba "cabecera truncada", que no dice nada de lo que pasa.
  if (datos.size() < 4) {
    *error = "'" + ruta + "': el archivo no llega ni a 4 bytes.";
    return false;
  }
  const uint32_t magic = LeerU32BE(datos.data());
  if (magic != 0x00000803u) {
    // 0x801 es el numero de las ETIQUETAS: confundir los dos archivos es el
    // error facil, porque los nombres solo se diferencian en una palabra.
    *error = "'" + ruta + "': numero magico 0x" + std::to_string(magic) +
             (magic == 0x00000801u
                  ? "; ese es el de las ETIQUETAS, no el de las imagenes."
                  : "; se esperaba 0x803 (imagenes IDX).");
    return false;
  }

  if (datos.size() < 16) {
    *error = "'" + ruta + "': cabecera truncada, hacen falta 16 bytes.";
    return false;
  }
  const uint32_t total = LeerU32BE(datos.data() + 4);
  const uint32_t filas = LeerU32BE(datos.data() + 8);
  const uint32_t cols = LeerU32BE(datos.data() + 12);
  const size_t esperados = static_cast<size_t>(total) * filas * cols;
  if (datos.size() != 16 + esperados) {
    *error = "'" + ruta + "': la cabecera anuncia " + std::to_string(total) + "x" +
             std::to_string(filas) + "x" + std::to_string(cols) + " = " +
             std::to_string(esperados) + " bytes y el archivo trae " +
             std::to_string(datos.size() - 16) + ".";
    return false;
  }

  *n = static_cast<int>(total);
  salida->Resize({static_cast<int>(total), 1, static_cast<int>(filas),
                  static_cast<int>(cols)});
  const float escala = normalizar ? (1.0f / 255.0f) : 1.0f;
  for (size_t i = 0; i < esperados; ++i) {
    (*salida)[i] = static_cast<float>(datos[16 + i]) * escala;
  }
  return true;
}

bool LeerIdxEtiquetas(const std::string& ruta, Tensor* salida, int* n, std::string* error) {
  std::vector<uint8_t> datos;
  if (!LeerArchivo(ruta, &datos, error)) return false;
  if (datos.size() < 4) {
    *error = "'" + ruta + "': el archivo no llega ni a 4 bytes.";
    return false;
  }
  const uint32_t magic = LeerU32BE(datos.data());
  if (magic != 0x00000801u) {
    *error = "'" + ruta + "': numero magico 0x" + std::to_string(magic) +
             (magic == 0x00000803u
                  ? "; ese es el de las IMAGENES, no el de las etiquetas."
                  : "; se esperaba 0x801 (etiquetas IDX).");
    return false;
  }

  if (datos.size() < 8) {
    *error = "'" + ruta + "': cabecera truncada, hacen falta 8 bytes.";
    return false;
  }
  const uint32_t total = LeerU32BE(datos.data() + 4);
  if (datos.size() != 8 + total) {
    *error = "'" + ruta + "': la cabecera anuncia " + std::to_string(total) +
             " etiquetas y el archivo trae " + std::to_string(datos.size() - 8) + ".";
    return false;
  }

  *n = static_cast<int>(total);
  salida->Resize({static_cast<int>(total)});
  for (uint32_t i = 0; i < total; ++i) {
    const uint8_t e = datos[8 + i];
    if (e > 9) {
      *error = "'" + ruta + "': la etiqueta " + std::to_string(i) + " vale " +
               std::to_string(static_cast<int>(e)) + ", fuera de [0, 9].";
      return false;
    }
    (*salida)[i] = static_cast<float>(e);
  }
  return true;
}

bool LeerMnist(const std::string& ruta_imagenes, const std::string& ruta_etiquetas,
               ConjuntoMnist* salida, std::string* error, bool normalizar) {
  int n_img = 0, n_lab = 0;
  if (!LeerIdxImagenes(ruta_imagenes, &salida->imagenes, &n_img, error, normalizar)) {
    return false;
  }
  if (!LeerIdxEtiquetas(ruta_etiquetas, &salida->etiquetas, &n_lab, error)) return false;

  // Si las cuentas no cuadran, entrenar seguiria funcionando con las etiquetas
  // corridas y el modelo no aprenderia, sin que nada fallara. Mejor abortar.
  if (n_img != n_lab) {
    *error = "hay " + std::to_string(n_img) + " imagenes y " + std::to_string(n_lab) +
             " etiquetas; no se pueden emparejar.";
    salida->n = 0;
    return false;
  }
  salida->n = n_img;
  return true;
}

}  // namespace data
}  // namespace neuralsuite
