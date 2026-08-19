// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parity_crnn.cpp
 * @brief Reproduce en C++ el CRNN de referencia exportado desde PyTorch.
 *
 * Carga el archivo NSPARITY que escribe tools/parity/export_crnn.py, copia esos
 * pesos en CRNNModel, repite el calculo sobre la misma imagen y vuelca sus
 * resultados para que tools/parity/compare_crnn.py los contraste.
 *
 * Es la verificacion fina del OCR. Las pruebas internas del CRNN son gruesas
 * por fuerza: la red es lineal a trozos y no hay paso de diferencia finita que
 * evite los codos. Aqui se compara gradiente contra gradiente.
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
using nsparity::Bundle;
using nsparity::Key;
using nsparity::ReadBundle;
using nsparity::Require;
using nsparity::WriteBundle;

int main(int argc, char** argv) {
  std::string in_path = "/tmp/crnn_ref.nsp";
  std::string out_path = "/tmp/crnn_cpp.nsp";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--in" && i + 1 < argc) in_path = argv[++i];
    if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
  }

  try {
    const Bundle ref = ReadBundle(in_path);
    const Array& meta = Require(ref, "meta");
    const int batch = static_cast<int>(meta.data[0]);
    const int height = static_cast<int>(meta.data[1]);
    const int width = static_cast<int>(meta.data[2]);
    const int hidden = static_cast<int>(meta.data[3]);
    const int steps = static_cast<int>(meta.data[4]);
    const int classes = static_cast<int>(meta.data[5]);

    CRNNModel crnn(1, hidden, classes);
    auto params = crnn.GetParameters();
    auto grads = crnn.GetGradients();

    const size_t expected = static_cast<size_t>(Require(ref, "num_params").data[0]);
    if (params.size() != expected) {
      std::cerr << "El modelo de C++ expone " << params.size() << " parametros y la referencia "
                << expected << ".\n";
      return 1;
    }

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

    Tensor x({batch, 1, height, width});
    std::memcpy(x.Data(), x_ref.data.data(), x_ref.data.size() * sizeof(float));

    Tensor out_t = crnn.Forward(x);
    if (out_t.Shape() != std::vector<int>({batch, steps, classes})) {
      std::cerr << "Forma de salida distinta: la referencia da [" << batch << ", " << steps << ", "
                << classes << "].\n";
      return 1;
    }

    double loss = 0.0;
    for (size_t i = 0; i < out_t.TotalSize(); ++i) {
      loss += static_cast<double>(out_t[i]) * w_ref.data[i];
    }

    // dL/dlogits = w, porque la perdida es la suma ponderada de las salidas.
    Tensor dout({batch, steps, classes});
    std::memcpy(dout.Data(), w_ref.data.data(), w_ref.data.size() * sizeof(float));
    Tensor dx = crnn.Backward(dout);

    Bundle bundle;
    Array loss_a; loss_a.shape = {1}; loss_a.data = {static_cast<float>(loss)};
    bundle["cpp_loss"] = loss_a;

    Array out_a; out_a.shape = {batch, steps, classes};
    out_a.data.assign(out_t.Data(), out_t.Data() + out_t.TotalSize());
    bundle["cpp_out"] = out_a;

    Array dx_a; dx_a.shape = {batch, 1, height, width};
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

    std::cout << "Pesos del CRNN de referencia cargados en CRNNModel de C++.\n";
    std::cout << "  loss C++ : " << loss << "\n";
    std::cout << "  escrito  : " << out_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
