// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file serialization.h
 * @brief Formato NSF: pesos con cabecera, version y comprobacion de integridad.
 *
 * El formato anterior volcaba los floats de todos los parametros uno tras otro,
 * sin cabecera. Cargar un archivo que no correspondia a la arquitectura
 * configurada no producia ningun error: el modelo se quedaba con datos
 * truncados o con basura y solo se notaba porque predecia mal. Tampoco habia
 * forma de saber que contenia un archivo, ni de detectar que estuviera cortado.
 *
 * NSF guarda cada tensor con su nombre y su forma, mas los metadatos de la
 * arquitectura, de modo que la carga puede rechazar lo que no encaja y decir
 * exactamente por que.
 *
 *     magic      8 bytes   "NSFMT001"
 *     version    uint32
 *     n_meta     uint32
 *       por entrada: clave y valor, cada uno uint32 de longitud + bytes
 *     n_tensors  uint32
 *       por tensor: nombre (uint32 + bytes), dtype uint32, rango uint32,
 *                   dims int32[rango], n_bytes uint64, datos
 *     checksum   uint32   suma de comprobacion de todo lo anterior
 *
 * Los enteros van en little-endian, que es el orden nativo de las plataformas
 * que soporta el proyecto; un archivo escrito en una maquina big-endian se
 * detectaria por el numero magico.
 *
 * **Este archivo se queda en la cabecera**: tiene plantillas, y el compilador
 * necesita su cuerpo alli donde se instancian.
 */

#ifndef NEURAL_SUITE_INCLUDE_SERIALIZATION_H_
#define NEURAL_SUITE_INCLUDE_SERIALIZATION_H_

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "parameter.h"
#include "tensor.h"

namespace neuralsuite {
namespace nsf {

inline constexpr char kMagic[8] = {'N', 'S', 'F', 'M', 'T', '0', '0', '1'};
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kDTypeFloat32 = 0;

/** @brief Resultado de una carga: exito, o el motivo exacto del fallo. */
struct Result {
  bool ok = false;
  std::string error;

  static Result Ok() { return {true, ""}; }
  static Result Fail(std::string msg) { return {false, std::move(msg)}; }
  explicit operator bool() const { return ok; }
};

namespace detail {

/**
 * @brief Suma de comprobacion sencilla (FNV-1a de 32 bits).
 *
 * No es criptografica: solo detecta truncamientos y corrupcion accidental, que
 * es de lo que se trata aqui.
 */
class Checksum {
 public:
  void Update(const void* data, size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
      hash_ ^= p[i];
      hash_ *= 16777619u;
    }
  }
  [[nodiscard]] uint32_t Value() const { return hash_; }

 private:
  uint32_t hash_ = 2166136261u;
};

template <typename T>
void WritePod(std::ostream& out, Checksum& sum, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  sum.Update(&value, sizeof(T));
}

inline void WriteString(std::ostream& out, Checksum& sum, const std::string& s) {
  const uint32_t len = static_cast<uint32_t>(s.size());
  WritePod(out, sum, len);
  out.write(s.data(), len);
  sum.Update(s.data(), len);
}

template <typename T>
bool ReadPod(std::istream& in, Checksum& sum, T* value) {
  in.read(reinterpret_cast<char*>(value), sizeof(T));
  if (!in) return false;
  sum.Update(value, sizeof(T));
  return true;
}

inline bool ReadString(std::istream& in, Checksum& sum, std::string* s) {
  uint32_t len = 0;
  if (!ReadPod(in, sum, &len)) return false;
  if (len > (1u << 20)) return false;  // un nombre no puede ser tan largo
  s->assign(len, '\0');
  in.read(&(*s)[0], len);
  if (!in) return false;
  sum.Update(s->data(), len);
  return true;
}

}  // namespace detail

/** @brief Un tensor con nombre, tal como se guarda o se espera al cargar. */
struct NamedTensor {
  std::string name;
  Tensor* tensor;
};

/**
 * @brief Escribe los tensores y los metadatos en `path`.
 */
inline Result Save(const std::string& path, const std::vector<NamedTensor>& tensors,
                   const std::map<std::string, std::string>& metadata) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return Result::Fail("No se pudo abrir para escritura: " + path);

  detail::Checksum sum;
  out.write(kMagic, sizeof(kMagic));
  sum.Update(kMagic, sizeof(kMagic));
  detail::WritePod(out, sum, kVersion);

  detail::WritePod(out, sum, static_cast<uint32_t>(metadata.size()));
  for (const auto& [key, value] : metadata) {
    detail::WriteString(out, sum, key);
    detail::WriteString(out, sum, value);
  }

  detail::WritePod(out, sum, static_cast<uint32_t>(tensors.size()));
  for (const NamedTensor& nt : tensors) {
    detail::WriteString(out, sum, nt.name);
    detail::WritePod(out, sum, kDTypeFloat32);
    detail::WritePod(out, sum, static_cast<uint32_t>(nt.tensor->Shape().size()));
    for (int d : nt.tensor->Shape()) detail::WritePod(out, sum, static_cast<int32_t>(d));

    const uint64_t n_bytes = static_cast<uint64_t>(nt.tensor->TotalSize()) * sizeof(float);
    detail::WritePod(out, sum, n_bytes);
    out.write(reinterpret_cast<const char*>(nt.tensor->Data()),
              static_cast<std::streamsize>(n_bytes));
    sum.Update(nt.tensor->Data(), static_cast<size_t>(n_bytes));
  }

  const uint32_t checksum = sum.Value();
  out.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));

  if (!out) return Result::Fail("Error de escritura en: " + path);
  return Result::Ok();
}

/**
 * @brief Carga en `tensors` los pesos de `path`, comprobando que correspondan.
 *
 * Falla, indicando el motivo, si el archivo no es NSF, si su version no es la
 * soportada, si algun metadato esperado no coincide, si falta un tensor, si su
 * forma difiere, o si la suma de comprobacion no cuadra.
 */
inline Result Load(const std::string& path, const std::vector<NamedTensor>& tensors,
                   const std::map<std::string, std::string>& expected_metadata) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Result::Fail("No se pudo abrir para lectura: " + path);

  detail::Checksum sum;
  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  // El numero magico entra en la suma igual que al escribir; omitirlo aqui
  // haria que ningun archivo valido superase la comprobacion de integridad.
  sum.Update(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    return Result::Fail(
        path + ": no es un archivo NSF. Los pesos guardados con versiones "
               "anteriores no llevaban cabecera y no se pueden cargar; hay que "
               "volver a entrenar o reconvertirlos.");
  }

  uint32_t version = 0;
  if (!detail::ReadPod(in, sum, &version)) return Result::Fail(path + ": cabecera truncada.");
  if (version != kVersion) {
    return Result::Fail(path + ": version de formato " + std::to_string(version) +
                        ", esta compilacion soporta la " + std::to_string(kVersion) + ".");
  }

  uint32_t n_meta = 0;
  if (!detail::ReadPod(in, sum, &n_meta)) return Result::Fail(path + ": metadatos truncados.");
  std::map<std::string, std::string> metadata;
  for (uint32_t i = 0; i < n_meta; ++i) {
    std::string key, value;
    if (!detail::ReadString(in, sum, &key) || !detail::ReadString(in, sum, &value)) {
      return Result::Fail(path + ": metadatos truncados.");
    }
    metadata[key] = value;
  }

  // La comprobacion que da sentido al formato: el archivo debe corresponder a
  // la arquitectura que se esta construyendo.
  for (const auto& [key, expected] : expected_metadata) {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
      return Result::Fail(path + ": el archivo no declara '" + key + "'.");
    }
    if (it->second != expected) {
      return Result::Fail(path + ": " + key + " del archivo es " + it->second +
                          " y el modelo esperaba " + expected + ".");
    }
  }

  uint32_t n_tensors = 0;
  if (!detail::ReadPod(in, sum, &n_tensors)) return Result::Fail(path + ": tabla truncada.");
  if (n_tensors != tensors.size()) {
    return Result::Fail(path + ": contiene " + std::to_string(n_tensors) +
                        " tensores y el modelo tiene " + std::to_string(tensors.size()) + ".");
  }

  for (uint32_t i = 0; i < n_tensors; ++i) {
    std::string name;
    uint32_t dtype = 0, rank = 0;
    if (!detail::ReadString(in, sum, &name) || !detail::ReadPod(in, sum, &dtype) ||
        !detail::ReadPod(in, sum, &rank)) {
      return Result::Fail(path + ": tensor " + std::to_string(i) + " truncado.");
    }
    if (name != tensors[i].name) {
      return Result::Fail(path + ": el tensor " + std::to_string(i) + " se llama '" + name +
                          "' y el modelo espera '" + tensors[i].name + "'.");
    }
    if (dtype != kDTypeFloat32) {
      return Result::Fail(path + ": '" + name + "' no es float32.");
    }

    std::vector<int> dims(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      int32_t dim = 0;
      if (!detail::ReadPod(in, sum, &dim)) return Result::Fail(path + ": forma truncada.");
      dims[d] = dim;
    }
    if (dims != tensors[i].tensor->Shape()) {
      return Result::Fail(path + ": '" + name + "' tiene otra forma que la del modelo.");
    }

    uint64_t n_bytes = 0;
    if (!detail::ReadPod(in, sum, &n_bytes)) return Result::Fail(path + ": longitud truncada.");
    const uint64_t expected_bytes =
        static_cast<uint64_t>(tensors[i].tensor->TotalSize()) * sizeof(float);
    if (n_bytes != expected_bytes) {
      return Result::Fail(path + ": '" + name + "' declara " + std::to_string(n_bytes) +
                          " bytes y se esperaban " + std::to_string(expected_bytes) + ".");
    }

    in.read(reinterpret_cast<char*>(tensors[i].tensor->Data()),
            static_cast<std::streamsize>(n_bytes));
    if (!in) return Result::Fail(path + ": datos de '" + name + "' truncados.");
    sum.Update(tensors[i].tensor->Data(), static_cast<size_t>(n_bytes));
  }

  uint32_t stored = 0;
  in.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (!in) return Result::Fail(path + ": falta la suma de comprobacion.");
  if (stored != sum.Value()) {
    return Result::Fail(path + ": la suma de comprobacion no cuadra; el archivo esta corrupto.");
  }
  return Result::Ok();
}

/** @brief Convierte los parametros con nombre de un Module al formato del archivo. */
inline std::vector<NamedTensor> FromNamedParameters(
    const std::vector<std::pair<std::string, Parameter*>>& named) {
  std::vector<NamedTensor> out;
  out.reserve(named.size());
  for (const auto& [name, param] : named) {
    out.push_back({name, &param->Value()});
  }
  return out;
}

}  // namespace nsf
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_SERIALIZATION_H_
