// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file nsparity.h
 * @brief Lectura y escritura del formato NSPARITY de intercambio con PyTorch.
 *
 * Un archivo NSPARITY es una bolsa de arrays float32 con nombre: la cabecera
 * `NSPARITY`, la version, cuantos arrays vienen, y para cada uno su nombre, sus
 * dimensiones y sus datos. Basta para mover pesos, entradas y gradientes entre
 * los dos lenguajes sin depender de ninguna biblioteca.
 *
 * Vive aqui, y no dentro de cada programa de paridad, porque cada nuevo caso
 * que se compara —GPT, LSTM, BiLSTM— repetia estas mismas noventa lineas. El
 * equivalente en Python es tools/parity/nsparity.py; los dos deben leer y
 * escribir exactamente lo mismo.
 */

#ifndef NEURAL_SUITE_TOOLS_PARITY_NSPARITY_H_
#define NEURAL_SUITE_TOOLS_PARITY_NSPARITY_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nsparity {

struct Array {
  std::vector<int> shape;
  std::vector<float> data;
  size_t Count() const {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return shape.empty() ? 0 : n;
  }
};

using Bundle = std::map<std::string, Array>;

template <typename T>
T ReadPod(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

inline Bundle ReadBundle(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("No se pudo abrir " + path);
  char magic[8];
  in.read(magic, 8);
  if (std::memcmp(magic, "NSPARITY", 8) != 0) {
    throw std::runtime_error(path + ": no es un archivo NSPARITY");
  }
  if (ReadPod<int32_t>(in) != 1) throw std::runtime_error(path + ": version no soportada");
  const int32_t count = ReadPod<int32_t>(in);

  Bundle bundle;
  for (int32_t i = 0; i < count; ++i) {
    const int32_t name_len = ReadPod<int32_t>(in);
    std::string name(static_cast<size_t>(name_len), '\0');
    in.read(name.data(), name_len);
    Array array;
    const int32_t ndim = ReadPod<int32_t>(in);
    for (int32_t d = 0; d < ndim; ++d) array.shape.push_back(ReadPod<int32_t>(in));
    array.data.resize(array.Count());
    in.read(reinterpret_cast<char*>(array.data.data()),
            static_cast<std::streamsize>(array.data.size() * sizeof(float)));
    bundle.emplace(std::move(name), std::move(array));
  }
  if (!in) throw std::runtime_error(path + ": archivo truncado");
  return bundle;
}

inline void WriteBundle(const std::string& path, const Bundle& bundle) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("No se pudo escribir " + path);
  out.write("NSPARITY", 8);
  const int32_t version = 1;
  const int32_t count = static_cast<int32_t>(bundle.size());
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& [name, array] : bundle) {
    const int32_t name_len = static_cast<int32_t>(name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    out.write(name.data(), name_len);
    const int32_t ndim = static_cast<int32_t>(array.shape.size());
    out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int d : array.shape) {
      const int32_t dim = d;
      out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    out.write(reinterpret_cast<const char*>(array.data.data()),
              static_cast<std::streamsize>(array.data.size() * sizeof(float)));
  }
}

inline const Array& Require(const Bundle& b, const std::string& name) {
  auto it = b.find(name);
  if (it == b.end()) throw std::runtime_error("Falta el tensor '" + name + "'");
  return it->second;
}

/** @brief Nombre indexado de la bolsa, del estilo `cpp_grad007`. */
inline std::string Key(const char* prefix, size_t i) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%s%03zu", prefix, i);
  return buf;
}

}  // namespace nsparity

#endif  // NEURAL_SUITE_TOOLS_PARITY_NSPARITY_H_
