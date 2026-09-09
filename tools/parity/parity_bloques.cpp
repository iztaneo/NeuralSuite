// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parity_bloques.cpp
 * @brief Reproduce en C++ el RMSNorm y el SiLU de PyTorch.
 *
 * Carga el archivo NSPARITY que escribe tools/parity/export_bloques.py, usa esos
 * mismos numeros de entrada y vuelca los resultados para que
 * tools/parity/compare_bloques.py los contraste.
 *
 * Un gradient check ya confirma que el backward deriva el forward escrito. Lo
 * que esto anade es que ese forward sea de verdad un RMSNorm y no un LayerNorm
 * disfrazado: la diferencia esta en restar o no la media, y una implementacion
 * que la restara seria coherente consigo misma y pasaria el gradient check.
 */

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "neuralsuite.h"
#include "nsparity.h"

using namespace neuralsuite;
using nsparity::Bundle;
using nsparity::ReadBundle;
using nsparity::Require;
using nsparity::WriteBundle;

namespace {

Tensor ATensor(const nsparity::Array& a) {
  std::vector<int> forma(a.shape.begin(), a.shape.end());
  Tensor t(forma);
  std::memcpy(t.Data(), a.data.data(), a.data.size() * sizeof(float));
  return t;
}

nsparity::Array AArray(const Tensor& t) {
  nsparity::Array a;
  a.shape.assign(t.Shape().begin(), t.Shape().end());
  a.data.assign(t.Data(), t.Data() + t.TotalSize());
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  std::string entrada, salida;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--in" && i + 1 < argc) entrada = argv[++i];
    else if (a == "--out" && i + 1 < argc) salida = argv[++i];
  }
  if (entrada.empty() || salida.empty()) {
    std::cerr << "Uso: parity_bloques --in ref.nsp --out cpp.nsp\n";
    return 1;
  }

  const Bundle ref = ReadBundle(entrada);
  const Tensor x = ATensor(Require(ref, "x"));
  const Tensor w = ATensor(Require(ref, "w"));
  const Tensor gamma = ATensor(Require(ref, "gamma"));
  const nsparity::Array& meta = Require(ref, "meta");
  const int ancho = static_cast<int>(meta.data[1]);

  Bundle out;

  // --- RMSNorm
  RMSNormLayer rms(ancho);
  std::memcpy(rms.Gamma().Data(), gamma.Data(), gamma.TotalSize() * sizeof(float));
  const Tensor y_rms = rms.Forward(x);
  const Tensor dx_rms = rms.Backward(w);
  out["rms_y"] = AArray(y_rms);
  out["rms_dx"] = AArray(dx_rms);
  out["rms_dgamma"] = AArray(*rms.GetGradients()[0]);

  // --- SiLU, sobre la misma entrada
  Tensor y_silu, dx_silu;
  SiluForward(x, y_silu);
  SiluBackward(w, x, dx_silu);
  out["silu_y"] = AArray(y_silu);
  out["silu_dx"] = AArray(dx_silu);

  // --- GroupNorm
  {
    const Tensor gx = ATensor(Require(ref, "gn_x"));
    const Tensor gw = ATensor(Require(ref, "gn_w"));
    const Tensor ggamma = ATensor(Require(ref, "gn_gamma"));
    const Tensor gbeta = ATensor(Require(ref, "gn_beta"));
    const nsparity::Array& gmeta = Require(ref, "gn_meta");
    const int canales = static_cast<int>(gmeta.data[1]);
    const int grupos = static_cast<int>(gmeta.data[4]);

    GroupNormLayer gn(grupos, canales);
    std::memcpy(gn.Gamma().Data(), ggamma.Data(), ggamma.TotalSize() * sizeof(float));
    std::memcpy(gn.Beta().Data(), gbeta.Data(), gbeta.TotalSize() * sizeof(float));
    const Tensor gy = gn.Forward(gx);
    const Tensor gdx = gn.Backward(gw);
    out["gn_y"] = AArray(gy);
    out["gn_dx"] = AArray(gdx);
    out["gn_dgamma"] = AArray(*gn.GetGradients()[0]);
    out["gn_dbeta"] = AArray(*gn.GetGradients()[1]);
  }

  // --- Upsample2D y Downsample2D
  {
    const Tensor ux = ATensor(Require(ref, "up_x"));
    const Tensor uw = ATensor(Require(ref, "up_w"));
    Upsample2D up(2);
    out["up_y"] = AArray(up.Forward(ux));
    out["up_dx"] = AArray(up.Backward(uw));

    const Tensor dxin = ATensor(Require(ref, "dn_x"));
    const Tensor dw = ATensor(Require(ref, "dn_w"));
    Downsample2D dn(2);
    out["dn_y"] = AArray(dn.Forward(dxin));
    out["dn_dx"] = AArray(dn.Backward(dw));
  }

  // --- CrossAttention
  {
    const nsparity::Array& cm = Require(ref, "ca_meta");
    const int E = static_cast<int>(cm.data[0]);
    const int Hh = static_cast<int>(cm.data[1]);

    CrossAttention ca(E, Hh);
    auto copiar = [](Linear& capa, const Tensor& w, const Tensor& b) {
      std::memcpy(capa.Weight().Data(), w.Data(), w.TotalSize() * sizeof(float));
      std::memcpy(capa.Bias().Data(), b.Data(), b.TotalSize() * sizeof(float));
    };
    copiar(ca.QProj(), ATensor(Require(ref, "ca_Wq")), ATensor(Require(ref, "ca_bq")));
    copiar(ca.KProj(), ATensor(Require(ref, "ca_Wk")), ATensor(Require(ref, "ca_bk")));
    copiar(ca.VProj(), ATensor(Require(ref, "ca_Wv")), ATensor(Require(ref, "ca_bv")));
    copiar(ca.OProj(), ATensor(Require(ref, "ca_Wo")), ATensor(Require(ref, "ca_bo")));

    const Tensor qx = ATensor(Require(ref, "ca_q"));
    const Tensor cx = ATensor(Require(ref, "ca_ctx"));
    const Tensor wx = ATensor(Require(ref, "ca_w"));

    out["ca_y"] = AArray(ca.Forward(qx, cx));
    out["ca_dq"] = AArray(ca.Backward(wx));
    out["ca_dctx"] = AArray(ca.GradContexto());
  }

  WriteBundle(salida, out);
  std::cout << "Escrito " << salida << "\n";
  return 0;
}
