// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parity_lstm.cpp
 * @brief Reproduce en C++ el forward y el backward de un nn.LSTM de PyTorch.
 *
 * Carga el archivo NSPARITY que escribe tools/parity/export_lstm.py, copia esos
 * pesos en la capa LSTM, repite el calculo sobre la misma entrada y vuelca sus
 * resultados para que tools/parity/compare_lstm.py los contraste.
 */

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

const Array& Require(const Bundle& b, const std::string& name) {
  auto it = b.find(name);
  if (it == b.end()) throw std::runtime_error("Falta el tensor '" + name + "'");
  return it->second;
}

}  // namespace

int main(int argc, char** argv) {
  std::string in_path = "/tmp/lstm_ref.nsp";
  std::string out_path = "/tmp/lstm_cpp.nsp";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--in" && i + 1 < argc) in_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  try {
    const Bundle ref = ReadBundle(in_path);
    const Array& meta = Require(ref, "meta");
    const int seq = static_cast<int>(meta.data[0]);
    const int batch = static_cast<int>(meta.data[1]);
    const int input_size = static_cast<int>(meta.data[2]);
    const int hidden = static_cast<int>(meta.data[3]);

    LSTM lstm(input_size, hidden);
    auto params = lstm.GetParameters();
    auto grads = lstm.GetGradients();

    for (size_t i = 0; i < params.size(); ++i) {
      char key[16];
      std::snprintf(key, sizeof(key), "param%03zu", i);
      const Array& src = Require(ref, key);
      if (src.Count() != params[i]->TotalSize()) {
        std::cerr << "Tamano distinto en el parametro " << i << ": referencia " << src.Count()
                  << ", C++ " << params[i]->TotalSize() << ".\n";
        return 1;
      }
      std::memcpy(params[i]->Data(), src.data.data(), src.data.size() * sizeof(float));
    }

    const Array& x_ref = Require(ref, "x");
    const Array& w_ref = Require(ref, "w");

    Tensor x({seq, batch, input_size});
    std::memcpy(x.Data(), x_ref.data.data(), x_ref.data.size() * sizeof(float));

    Tensor out_t = lstm.Forward(x);

    double loss = 0.0;
    for (size_t i = 0; i < out_t.TotalSize(); ++i) {
      loss += static_cast<double>(out_t[i]) * w_ref.data[i];
    }

    // dL/dh = w, porque la perdida es la suma ponderada de las salidas.
    Tensor dout({seq, batch, hidden});
    std::memcpy(dout.Data(), w_ref.data.data(), w_ref.data.size() * sizeof(float));
    Tensor dx = lstm.Backward(dout);

    Bundle bundle;
    Array loss_a; loss_a.shape = {1}; loss_a.data = {static_cast<float>(loss)};
    bundle["cpp_loss"] = loss_a;

    Array out_a; out_a.shape = {seq, batch, hidden};
    out_a.data.assign(out_t.Data(), out_t.Data() + out_t.TotalSize());
    bundle["cpp_out"] = out_a;

    Array dx_a; dx_a.shape = {seq, batch, input_size};
    dx_a.data.assign(dx.Data(), dx.Data() + dx.TotalSize());
    bundle["cpp_dx"] = dx_a;

    for (size_t i = 0; i < grads.size(); ++i) {
      char key[20];
      std::snprintf(key, sizeof(key), "cpp_grad%03zu", i);
      Array g; g.shape = grads[i]->Shape();
      g.data.assign(grads[i]->Data(), grads[i]->Data() + grads[i]->TotalSize());
      bundle[key] = g;
    }
    Array n; n.shape = {1}; n.data = {static_cast<float>(grads.size())};
    bundle["num_grads"] = n;

    WriteBundle(out_path, bundle);

    std::cout << "Pesos de nn.LSTM cargados en la capa LSTM de C++.\n";
    std::cout << "  loss C++ : " << loss << "\n";
    std::cout << "  escrito  : " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
