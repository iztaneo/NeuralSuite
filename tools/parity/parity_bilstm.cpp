// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parity_bilstm.cpp
 * @brief Reproduce en C++ un nn.LSTM(bidirectional=True) de PyTorch.
 *
 * Carga el archivo NSPARITY que escribe tools/parity/export_bilstm.py, copia
 * esos pesos en la capa BiLSTM, repite el calculo sobre la misma entrada y
 * vuelca sus resultados para que tools/parity/compare_bilstm.py los contraste.
 *
 * Es la comprobacion que ninguna prueba interna puede dar: el gradient check
 * confirma que el backward deriva el forward escrito, y la prueba de
 * direccionalidad confirma de que depende cada salida, pero solo PyTorch dice
 * si los numeros son los que deberian ser.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "neuralsuite.h"
#include "nsparity.h"

using namespace neuralsuite;
using nsparity::Array;
using nsparity::Key;
using nsparity::Bundle;
using nsparity::ReadBundle;
using nsparity::Require;
using nsparity::WriteBundle;

int main(int argc, char** argv) {
  std::string in_path = "/tmp/bilstm_ref.nsp";
  std::string out_path = "/tmp/bilstm_cpp.nsp";
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

    BiLSTM bilstm(input_size, hidden);
    auto params = bilstm.GetParameters();
    auto grads = bilstm.GetGradients();

    // El exportador escribe los parametros en el orden de registro de la capa:
    // primero los cuatro del sentido directo, luego los cuatro del inverso.
    for (size_t i = 0; i < params.size(); ++i) {
      const Array& src = Require(ref, Key("param", i));
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

    Tensor out_t = bilstm.Forward(x);

    double loss = 0.0;
    for (size_t i = 0; i < out_t.TotalSize(); ++i) {
      loss += static_cast<double>(out_t[i]) * w_ref.data[i];
    }

    // dL/dh = w, porque la perdida es la suma ponderada de las salidas.
    Tensor dout({seq, batch, 2 * hidden});
    std::memcpy(dout.Data(), w_ref.data.data(), w_ref.data.size() * sizeof(float));
    Tensor dx = bilstm.Backward(dout);

    Bundle bundle;
    Array loss_a; loss_a.shape = {1}; loss_a.data = {static_cast<float>(loss)};
    bundle["cpp_loss"] = loss_a;

    Array out_a; out_a.shape = {seq, batch, 2 * hidden};
    out_a.data.assign(out_t.Data(), out_t.Data() + out_t.TotalSize());
    bundle["cpp_out"] = out_a;

    Array dx_a; dx_a.shape = {seq, batch, input_size};
    dx_a.data.assign(dx.Data(), dx.Data() + dx.TotalSize());
    bundle["cpp_dx"] = dx_a;

    for (size_t i = 0; i < grads.size(); ++i) {
      Array g; g.shape = grads[i]->Shape();
      g.data.assign(grads[i]->Data(), grads[i]->Data() + grads[i]->TotalSize());
      bundle[Key("cpp_grad", i)] = g;
    }
    Array n; n.shape = {1}; n.data = {static_cast<float>(grads.size())};
    bundle["num_grads"] = n;

    WriteBundle(out_path, bundle);

    std::cout << "Pesos de nn.LSTM(bidirectional=True) cargados en BiLSTM de C++.\n";
    std::cout << "  loss C++ : " << loss << "\n";
    std::cout << "  escrito  : " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
