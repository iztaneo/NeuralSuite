// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parity_gpt.cpp
 * @brief Reproduce en C++ el forward y el backward exportados desde PyTorch.
 *
 * Carga un archivo NSPARITY generado por tools/parity/export_gpt.py, copia esos
 * pesos en un GPTModel, ejecuta el mismo forward y backward sobre la misma
 * entrada y vuelca sus resultados en otro archivo NSPARITY. La comparación
 * numérica la hace tools/parity/compare_gpt.py.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

namespace {

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

Bundle ReadBundle(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("No se pudo abrir " + path);

  char magic[8];
  in.read(magic, 8);
  if (std::memcmp(magic, "NSPARITY", 8) != 0) {
    throw std::runtime_error(path + ": no es un archivo NSPARITY");
  }
  const int32_t version = ReadPod<int32_t>(in);
  if (version != 1) throw std::runtime_error(path + ": version no soportada");
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

void WriteBundle(const std::string& path, const Bundle& bundle) {
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

const Array& Require(const Bundle& bundle, const std::string& name) {
  auto it = bundle.find(name);
  if (it == bundle.end()) throw std::runtime_error("Falta el tensor '" + name + "'");
  return it->second;
}

std::string Key(const char* prefix, size_t i) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%s%03zu", prefix, i);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path = "/tmp/gpt_ref.nsp";
  std::string out_path = "/tmp/gpt_cpp.nsp";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--in" && i + 1 < argc) in_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  try {
    const Bundle ref = ReadBundle(in_path);
    const Array& meta = Require(ref, "meta");

    GPTConfig cfg;
    cfg.vocab_size = static_cast<int>(meta.data[0]);
    cfg.block_size = static_cast<int>(meta.data[1]);
    cfg.n_layer = static_cast<int>(meta.data[2]);
    cfg.n_head = static_cast<int>(meta.data[3]);
    cfg.n_embd = static_cast<int>(meta.data[4]);
    const int batch = static_cast<int>(meta.data[5]);
    const int seq = static_cast<int>(meta.data[6]);

    GPTModel model(cfg);
    auto params = model.GetParameters();
    auto grads = model.GetGradients();

    const size_t expected = static_cast<size_t>(Require(ref, "num_params").data[0]);
    if (params.size() != expected) {
      std::cerr << "El modelo de C++ expone " << params.size()
                << " parametros y el de referencia " << expected << ".\n";
      return 1;
    }

    // Copiar los pesos de referencia. El exportador ya los dejo en el orden de
    // GetParameters() y con las matrices densas transpuestas.
    for (size_t i = 0; i < params.size(); ++i) {
      const Array& src = Require(ref, Key("param", i));
      if (src.Count() != params[i]->TotalSize()) {
        std::cerr << "Tamano distinto en el parametro " << i << ": referencia "
                  << src.Count() << ", C++ " << params[i]->TotalSize() << ".\n";
        return 1;
      }
      std::memcpy(params[i]->Data(), src.data.data(), src.data.size() * sizeof(float));
    }

    const Array& idx_ref = Require(ref, "idx");
    const Array& tgt_ref = Require(ref, "targets");

    Tensor idx({batch, seq});
    std::memcpy(idx.Data(), idx_ref.data.data(), idx_ref.data.size() * sizeof(float));
    Tensor targets({batch * seq});
    std::memcpy(targets.Data(), tgt_ref.data.data(), tgt_ref.data.size() * sizeof(float));

    Tensor logits = model.Forward(idx);

    Tensor logits_2d({batch * seq, cfg.vocab_size});
    std::memcpy(logits_2d.Data(), logits.Data(), logits.TotalSize() * sizeof(float));

    CrossEntropyLoss criterion;
    const float loss = criterion.Forward(logits_2d, targets);

    Tensor dlogits_2d = criterion.Backward();
    Tensor dlogits({batch, seq, cfg.vocab_size});
    std::memcpy(dlogits.Data(), dlogits_2d.Data(), dlogits_2d.TotalSize() * sizeof(float));
    model.Backward(dlogits);

    Bundle out;
    Array loss_array;
    loss_array.shape = {1};
    loss_array.data = {loss};
    out["cpp_loss"] = loss_array;

    Array logits_array;
    logits_array.shape = {batch, seq, cfg.vocab_size};
    logits_array.data.assign(logits.Data(), logits.Data() + logits.TotalSize());
    out["cpp_logits"] = logits_array;

    for (size_t i = 0; i < grads.size(); ++i) {
      Array g;
      g.shape = grads[i]->Shape();
      g.data.assign(grads[i]->Data(), grads[i]->Data() + grads[i]->TotalSize());
      out[Key("cpp_grad", i)] = g;
    }
    Array n;
    n.shape = {1};
    n.data = {static_cast<float>(grads.size())};
    out["num_grads"] = n;

    WriteBundle(out_path, out);

    std::cout << "Pesos de referencia cargados en el GPT de C++.\n";
    std::cout << "  parametros : " << params.size() << "\n";
    std::cout << "  loss C++   : " << loss << "\n";
    std::cout << "  escrito    : " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
