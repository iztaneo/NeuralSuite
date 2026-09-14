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
#include <memory>
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

  // --- SwiGLU
  {
    const nsparity::Array& sm = Require(ref, "sw_meta");
    const int D = static_cast<int>(sm.data[0]);
    const int Hh = static_cast<int>(sm.data[1]);

    SwiGLU sw(D, Hh);
    auto copiar = [](Linear& capa, const Tensor& w, const Tensor& b) {
      std::memcpy(capa.Weight().Data(), w.Data(), w.TotalSize() * sizeof(float));
      std::memcpy(capa.Bias().Data(), b.Data(), b.TotalSize() * sizeof(float));
    };
    copiar(sw.Gate(), ATensor(Require(ref, "sw_Wg")), ATensor(Require(ref, "sw_bg")));
    copiar(sw.Up(), ATensor(Require(ref, "sw_Wu")), ATensor(Require(ref, "sw_bu")));
    copiar(sw.Down(), ATensor(Require(ref, "sw_Wd")), ATensor(Require(ref, "sw_bd")));

    const Tensor sx = ATensor(Require(ref, "sw_x"));
    const Tensor sw_w = ATensor(Require(ref, "sw_w"));
    out["sw_y"] = AArray(sw.Forward(sx));
    out["sw_dx"] = AArray(sw.Backward(sw_w));
    out["sw_dWg"] = AArray(*sw.Gate().GetGradients()[0]);
    out["sw_dWu"] = AArray(*sw.Up().GetGradients()[0]);
    out["sw_dWd"] = AArray(*sw.Down().GetGradients()[0]);
  }

  // --- RoPE
  {
    const nsparity::Array& rm = Require(ref, "rope_meta");
    const int heads = static_cast<int>(rm.data[2]);
    const int pos0 = static_cast<int>(rm.data[4]);

    const Tensor rx = ATensor(Require(ref, "rope_x"));
    const Tensor rw = ATensor(Require(ref, "rope_w"));
    Tensor ry, rdx;
    RopeForward(rx, ry, heads, pos0);
    RopeBackward(rw, rdx, heads, pos0);
    out["rope_y"] = AArray(ry);
    out["rope_dx"] = AArray(rdx);
  }

  // --- DiffusionSchedule
  {
    const nsparity::Array& dm = Require(ref, "dif_meta");
    const int pasos = static_cast<int>(dm.data[0]);

    diffusion::DiffusionSchedule cal(pasos);
    out["dif_beta"] = AArray(cal.Beta());
    out["dif_alpha_bar"] = AArray(cal.AlphaBar());

    const Tensor x0 = ATensor(Require(ref, "dif_x0"));
    const Tensor ruido = ATensor(Require(ref, "dif_ruido"));
    const Tensor t = ATensor(Require(ref, "dif_pasos"));
    Tensor xt, x0_rec;
    cal.QSample(x0, ruido, t, &xt);
    cal.PredecirX0(xt, ruido, t, &x0_rec);
    out["dif_xt"] = AArray(xt);
    out["dif_x0_rec"] = AArray(x0_rec);
  }

  // --- TimeEmbedding
  {
    const nsparity::Array& tm = Require(ref, "te_meta");
    const int dim = static_cast<int>(tm.data[0]);
    const Tensor pasos = ATensor(Require(ref, "te_pasos"));
    Tensor emb;
    diffusion::TimeEmbedding(pasos, dim, &emb);
    out["te_emb"] = AArray(emb);
  }

  // --- ResBlockTiempo
  {
    const nsparity::Array& rm = Require(ref, "rb_meta");
    const int Cin = static_cast<int>(rm.data[1]);
    const int Cout = static_cast<int>(rm.data[2]);
    const int DT = static_cast<int>(rm.data[5]);
    const int G = static_cast<int>(rm.data[6]);

    diffusion::ResBlockTiempo rb(Cin, Cout, DT, G);
    auto cargar_norm = [&](GroupNormLayer& capa, const char* g, const char* b) {
      const Tensor tg = ATensor(Require(ref, g)), tb = ATensor(Require(ref, b));
      std::memcpy(capa.Gamma().Data(), tg.Data(), tg.TotalSize() * sizeof(float));
      std::memcpy(capa.Beta().Data(), tb.Data(), tb.TotalSize() * sizeof(float));
    };
    auto cargar_conv = [&](Conv2D& capa, const char* w, const char* b) {
      const Tensor tw = ATensor(Require(ref, w)), tb = ATensor(Require(ref, b));
      std::memcpy(capa.Weight().Data(), tw.Data(), tw.TotalSize() * sizeof(float));
      std::memcpy(capa.Bias().Data(), tb.Data(), tb.TotalSize() * sizeof(float));
    };
    cargar_norm(rb.Norm1(), "rb_n1_g", "rb_n1_b");
    cargar_norm(rb.Norm2(), "rb_n2_g", "rb_n2_b");
    cargar_conv(rb.Conv1(), "rb_c1_w", "rb_c1_b");
    cargar_conv(rb.Conv2(), "rb_c2_w", "rb_c2_b");
    cargar_conv(*rb.Atajo(), "rb_at_w", "rb_at_b");
    {
      const Tensor pw = ATensor(Require(ref, "rb_pt_w"));
      const Tensor pb = ATensor(Require(ref, "rb_pt_b"));
      std::memcpy(rb.ProyTiempo().Weight().Data(), pw.Data(),
                  pw.TotalSize() * sizeof(float));
      std::memcpy(rb.ProyTiempo().Bias().Data(), pb.Data(),
                  pb.TotalSize() * sizeof(float));
    }

    const Tensor rx = ATensor(Require(ref, "rb_x"));
    const Tensor rt = ATensor(Require(ref, "rb_t"));
    const Tensor rw = ATensor(Require(ref, "rb_w"));
    out["rb_y"] = AArray(rb.Forward(rx, rt));
    out["rb_dx"] = AArray(rb.Backward(rw));
    out["rb_dt"] = AArray(rb.GradTiempo());
  }

  // La U-Net sobrevive a su bloque: el caso de los muestreadores la reutiliza
  // con los mismos pesos, para que la comparacion mida el muestreo y no una
  // red distinta.
  std::unique_ptr<diffusion::UNet2D> unet_compartida;

  // --- UNet2D
  //
  // Aqui lo que se contrasta no son los numeros de cada capa —eso ya lo cubren
  // los casos de arriba— sino el cableado: el orden de los canales al
  // concatenar, que salto se une con que subida, y que el gradiente que vuelve
  // a cada salto sume los dos caminos que lo alcanzan. Los pesos entran por
  // nombre, submodulo a submodulo, para que un fallo senale al cableado y no a
  // un orden mal adivinado en la lista plana de parametros.
  {
    const nsparity::Array& um = Require(ref, "un_meta");
    const int Cu = static_cast<int>(um.data[1]);
    const int DTu = static_cast<int>(um.data[2]);
    const int Gu = static_cast<int>(um.data[3]);

    unet_compartida = std::make_unique<diffusion::UNet2D>(1, Cu, DTu, Gu);
    diffusion::UNet2D& unet = *unet_compartida;

    auto cargar_norm = [&](GroupNormLayer& capa, const std::string& g, const std::string& b) {
      const Tensor tg = ATensor(Require(ref, g)), tb = ATensor(Require(ref, b));
      std::memcpy(capa.Gamma().Data(), tg.Data(), tg.TotalSize() * sizeof(float));
      std::memcpy(capa.Beta().Data(), tb.Data(), tb.TotalSize() * sizeof(float));
    };
    auto cargar_conv = [&](Conv2D& capa, const std::string& w, const std::string& b) {
      const Tensor tw = ATensor(Require(ref, w)), tb = ATensor(Require(ref, b));
      std::memcpy(capa.Weight().Data(), tw.Data(), tw.TotalSize() * sizeof(float));
      std::memcpy(capa.Bias().Data(), tb.Data(), tb.TotalSize() * sizeof(float));
    };
    auto cargar_lin = [&](Linear& capa, const std::string& w, const std::string& b) {
      const Tensor tw = ATensor(Require(ref, w)), tb = ATensor(Require(ref, b));
      std::memcpy(capa.Weight().Data(), tw.Data(), tw.TotalSize() * sizeof(float));
      std::memcpy(capa.Bias().Data(), tb.Data(), tb.TotalSize() * sizeof(float));
    };
    auto cargar_bloque = [&](diffusion::ResBlockTiempo& rb, const std::string& p) {
      cargar_norm(rb.Norm1(), p + "_n1_w", p + "_n1_b");
      cargar_norm(rb.Norm2(), p + "_n2_w", p + "_n2_b");
      cargar_conv(rb.Conv1(), p + "_c1_w", p + "_c1_b");
      cargar_conv(rb.Conv2(), p + "_c2_w", p + "_c2_b");
      cargar_lin(rb.ProyTiempo(), p + "_pt_w", p + "_pt_b");
      // El atajo solo existe cuando cambian los canales; la referencia tampoco
      // lo exporta en ese caso, asi que las dos condiciones tienen que coincidir
      // o el Require de abajo lo delata.
      if (rb.Atajo() != nullptr) cargar_conv(*rb.Atajo(), p + "_at_w", p + "_at_b");
    };

    cargar_conv(unet.ConvEntrada(), "un_ce_w", "un_ce_b");
    cargar_conv(unet.ConvSalida(), "un_cs_w", "un_cs_b");
    cargar_norm(unet.NormSalida(), "un_ns_w", "un_ns_b");
    cargar_bloque(unet.ResBaja0(), "un_b0");
    cargar_bloque(unet.ResBaja1(), "un_b1");
    cargar_bloque(unet.ResCentro(), "un_ct");
    cargar_bloque(unet.ResAlta1(), "un_a1");
    cargar_bloque(unet.ResAlta0(), "un_a0");

    const Tensor ux = ATensor(Require(ref, "un_x"));
    const Tensor up = ATensor(Require(ref, "un_pasos"));
    const Tensor uw = ATensor(Require(ref, "un_w"));
    out["un_y"] = AArray(unet.Forward(ux, up));
    out["un_dx"] = AArray(unet.Backward(uw));
  }

  // --- DDPMSampler y DDIMSampler
  //
  // Es lo unico que fija la TRAYECTORIA. La prueba del oraculo analitico
  // aterriza en la imagen correcta aunque el camino sea otro —el oraculo se
  // autocorrige en cada paso— y de cuatro mutaciones deliberadas solo caza una.
  // Comparar la trayectoria completa contra PyTorch con el mismo ruido si
  // distingue las otras tres, porque cada una desvia los x intermedios.
  {
    const nsparity::Array& sm = Require(ref, "sm_meta");
    const int T = static_cast<int>(sm.data[0]);
    const int NP = static_cast<int>(sm.data[3]);

    diffusion::DiffusionSchedule cal(T, 1e-4f, 0.02f);
    const Tensor xT = ATensor(Require(ref, "sm_xT"));
    const Tensor ruidos = ATensor(Require(ref, "sm_ruido"));
    const size_t por_paso = xT.TotalSize();

    // El ruido no se genera aqui: se consume el exportado, indexado por paso.
    // Si cada lado generase el suyo, dos implementaciones correctas darian
    // resultados distintos y la comparacion no diria nada.
    auto fuente = [&](int paso, Tensor* destino) {
      std::memcpy(destino->Data(), ruidos.Data() + static_cast<size_t>(paso) * por_paso,
                  por_paso * sizeof(float));
    };
    diffusion::Predictor pred = [&](const Tensor& x, const Tensor& t) {
      return unet_compartida->Forward(x, t);
    };

    diffusion::DDPMSampler ddpm(cal);
    out["sm_ddpm"] = AArray(ddpm.Muestrear(pred, xT, fuente));

    diffusion::DDIMSampler ddim0(cal, NP, 0.0f);
    out["sm_ddim0"] = AArray(ddim0.Muestrear(pred, xT, fuente));

    diffusion::DDIMSampler ddim1(cal, NP, 1.0f);
    out["sm_ddim1"] = AArray(ddim1.Muestrear(pred, xT, fuente));

    diffusion::DDIMSampler ddim_full(cal, T, 1.0f);
    out["sm_ddim_full1"] = AArray(ddim_full.Muestrear(pred, xT, fuente));

    // La ruta con recorte de x0, que hasta ahora no tenia paridad. En DDIM fue
    // donde estaba el bug de no recalcular eps tras recortar.
    diffusion::DDPMSampler ddpm_rec(cal);
    ddpm_rec.RecortarX0(true);
    out["sm_ddpm_rec"] = AArray(ddpm_rec.Muestrear(pred, xT, fuente));
    diffusion::DDIMSampler ddim0_rec(cal, NP, 0.0f);
    ddim0_rec.RecortarX0(true);
    out["sm_ddim0_rec"] = AArray(ddim0_rec.Muestrear(pred, xT, fuente));
  }

  // --- ResBlock2D (codificador y decodificador del LDM-1)
  {
    const nsparity::Array& m = Require(ref, "r2_meta");
    const int Cin = static_cast<int>(m.data[1]);
    const int Cout = static_cast<int>(m.data[2]);
    const int G = static_cast<int>(m.data[5]);
    ResBlock2D rb(Cin, Cout, G);
    auto copiar = [&](Tensor& destino, const char* nombre) {
      const Tensor t = ATensor(Require(ref, nombre));
      if (t.TotalSize() != destino.TotalSize()) {
        throw std::runtime_error(std::string("r2: tamano distinto en ") + nombre);
      }
      std::memcpy(destino.Data(), t.Data(), t.TotalSize() * sizeof(float));
    };
    copiar(rb.Norm1().Gamma(), "r2_n1_g"); copiar(rb.Norm1().Beta(), "r2_n1_b");
    copiar(rb.Norm2().Gamma(), "r2_n2_g"); copiar(rb.Norm2().Beta(), "r2_n2_b");
    copiar(rb.Conv1().Weight(), "r2_c1_w"); copiar(rb.Conv1().Bias(), "r2_c1_b");
    copiar(rb.Conv2().Weight(), "r2_c2_w"); copiar(rb.Conv2().Bias(), "r2_c2_b");
    copiar(rb.Atajo()->Weight(), "r2_at_w"); copiar(rb.Atajo()->Bias(), "r2_at_b");

    rb.ZeroGrad();
    out["r2_y"] = AArray(rb.Forward(ATensor(Require(ref, "r2_x"))));
    out["r2_dx"] = AArray(rb.Backward(ATensor(Require(ref, "r2_w"))));
    out["r2_dc1w"] = AArray(*rb.Conv1().GetGradients()[0]);   // weight se registra primero
    out["r2_dn2g"] = AArray(*rb.Norm2().GetGradients()[0]);   // gamma se registra primero
  }

  // --- Codificador y Decodificador del LDM-1: el ensamblaje
  {
    const nsparity::Array& m = Require(ref, "ae_meta");
    const int C = static_cast<int>(m.data[1]);
    const int CL = static_cast<int>(m.data[2]);
    const int G = static_cast<int>(m.data[3]);
    auto copiar = [&](Tensor& destino, const std::string& nombre) {
      const Tensor t = ATensor(Require(ref, nombre));
      if (t.TotalSize() != destino.TotalSize()) {
        throw std::runtime_error("ae: tamano distinto en " + nombre);
      }
      std::memcpy(destino.Data(), t.Data(), t.TotalSize() * sizeof(float));
    };
    auto cargar_conv = [&](Conv2D& c, const std::string& p) {
      copiar(c.Weight(), p + "_w"); copiar(c.Bias(), p + "_b");
    };
    auto cargar_norm = [&](GroupNormLayer& n, const std::string& p) {
      copiar(n.Gamma(), p + "_w"); copiar(n.Beta(), p + "_b");
    };
    auto cargar_rb = [&](ResBlock2D& rb, const std::string& p) {
      cargar_norm(rb.Norm1(), p + "_n1"); cargar_conv(rb.Conv1(), p + "_c1");
      cargar_norm(rb.Norm2(), p + "_n2"); cargar_conv(rb.Conv2(), p + "_c2");
      if (rb.Atajo() != nullptr) cargar_conv(*rb.Atajo(), p + "_at");
    };

    latent::Codificador cod(1, C, CL, G);
    cargar_conv(cod.ConvEntrada(), "ae_c_ce_x");
    cargar_rb(cod.Res0(), "ae_c_r0"); cargar_rb(cod.Res1(), "ae_c_r1"); cargar_rb(cod.Res2(), "ae_c_r2");
    cargar_norm(cod.NormSalida(), "ae_c_ns_x");
    cargar_conv(cod.ConvSalida(), "ae_c_cs_x");
    cod.ZeroGrad();
    out["ae_cy"] = AArray(cod.Forward(ATensor(Require(ref, "ae_cx"))));
    out["ae_cdx"] = AArray(cod.Backward(ATensor(Require(ref, "ae_cw"))));
    out["ae_cdce"] = AArray(*cod.ConvEntrada().GetGradients()[0]);

    latent::Decodificador dec(CL, C, 1, G);
    cargar_conv(dec.ConvEntrada(), "ae_d_ce_x");
    cargar_rb(dec.Res0(), "ae_d_r0"); cargar_rb(dec.Res1(), "ae_d_r1"); cargar_rb(dec.Res2(), "ae_d_r2");
    cargar_norm(dec.NormSalida(), "ae_d_ns_x");
    cargar_conv(dec.ConvSalida(), "ae_d_cs_x");
    dec.ZeroGrad();
    out["ae_dy"] = AArray(dec.Forward(ATensor(Require(ref, "ae_dz"))));
    out["ae_ddz"] = AArray(dec.Backward(ATensor(Require(ref, "ae_dw"))));
    out["ae_ddce"] = AArray(*dec.ConvEntrada().GetGradients()[0]);
  }

  WriteBundle(salida, out);
  std::cout << "Escrito " << salida << "\n";
  return 0;
}
