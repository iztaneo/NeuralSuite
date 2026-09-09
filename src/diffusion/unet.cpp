// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de diffusion/unet.h.

#include "diffusion/unet.h"

#include <iostream>
#include <map>
#include "serialization.h"
#include <stdexcept>
#include <string>

#include <cstring>

#include "diffusion/time_embedding.h"

namespace neuralsuite {
namespace diffusion {

UNet2D::UNet2D(int canales_imagen, int canales, int dim_t, int grupos)
    : canales_imagen_(canales_imagen),
      canales_(canales),
      dim_t_(dim_t),
      grupos_(grupos),
      conv_entrada_(canales_imagen, canales, 3, 1, 1),
      res_baja0_(canales, canales, dim_t, grupos),
      res_baja1_(canales, 2 * canales, dim_t, grupos),
      res_centro_(2 * canales, 2 * canales, dim_t, grupos),
      // Al subir, la entrada es la concatenacion del mapa que sube con el
      // salto guardado, asi que los canales de entrada son la suma de ambos.
      res_alta1_(2 * canales + 2 * canales, canales, dim_t, grupos),
      res_alta0_(canales + canales, canales, dim_t, grupos),
      bajar0_(2), bajar1_(2), subir1_(2), subir0_(2),
      norm_salida_(grupos, canales),
      conv_salida_(canales, canales_imagen, 3, 1, 1) {
  // El orden de registro es el orden de `Parameters()` y el de los nombres en
  // el archivo de pesos. Solo se registra lo que tiene parametros: el
  // remuestreo no tiene ninguno.
  Register(&conv_entrada_, "conv_entrada");
  Register(&res_baja0_, "res_baja0");
  Register(&res_baja1_, "res_baja1");
  Register(&res_centro_, "res_centro");
  Register(&res_alta1_, "res_alta1");
  Register(&res_alta0_, "res_alta0");
  Register(&norm_salida_, "norm_salida");
  Register(&conv_salida_, "conv_salida");
}

Tensor UNet2D::Forward(const Tensor& x, const Tensor& pasos) {
  if (x.Shape().size() != 4) {
    throw std::invalid_argument("UNet2D: la entrada debe ser [N, C, H, W]");
  }
  // Dos bajadas de factor 2 seguidas de dos subidas solo devuelven la
  // resolucion original si esta es divisible por cuatro. `Downsample2D` lo
  // detectaria a mitad de camino; decirlo aqui nombra la condicion real de la
  // arquitectura en vez de la de una capa suelta.
  if (x.Shape()[2] % 4 != 0 || x.Shape()[3] % 4 != 0) {
    throw std::invalid_argument(
        "UNet2D: alto y ancho deben ser multiplos de 4 (dos bajadas de factor "
        "2); llegaron " + std::to_string(x.Shape()[2]) + "x" +
        std::to_string(x.Shape()[3]));
  }
  if (x.Shape()[1] != canales_imagen_) {
    throw std::invalid_argument(
        "UNet2D: se construyo para " + std::to_string(canales_imagen_) +
        " canal(es) y llegaron " + std::to_string(x.Shape()[1]));
  }
  n_ = x.Shape()[0];
  h_ = x.Shape()[2];
  w_ = x.Shape()[3];

  TimeEmbedding(pasos, dim_t_, &t_emb_);

  // Bajada, guardando lo que hara falta al subir.
  Tensor h = conv_entrada_.Forward(x);
  salto0_ = res_baja0_.Forward(h, t_emb_);          // [N, C, H, W]
  Tensor d0 = bajar0_.Forward(salto0_);             // [N, C, H/2, W/2]
  salto1_ = res_baja1_.Forward(d0, t_emb_);         // [N, 2C, H/2, W/2]
  Tensor d1 = bajar1_.Forward(salto1_);             // [N, 2C, H/4, W/4]

  Tensor centro = res_centro_.Forward(d1, t_emb_);  // [N, 2C, H/4, W/4]

  // Subida, concatenando los saltos. Sin ellos, el detalle fino perdido al
  // bajar habria que reconstruirlo desde cero.
  Tensor u1 = subir1_.Forward(centro);              // [N, 2C, H/2, W/2]
  Tensor c1 = Concat(u1, salto1_, /*eje=*/1);       // [N, 4C, H/2, W/2]
  Tensor a1 = res_alta1_.Forward(c1, t_emb_);       // [N, C, H/2, W/2]

  Tensor u0 = subir0_.Forward(a1);                  // [N, C, H, W]
  Tensor c0 = Concat(u0, salto0_, /*eje=*/1);       // [N, 2C, H, W]
  Tensor a0 = res_alta0_.Forward(c0, t_emb_);       // [N, C, H, W]

  pre_salida_ = norm_salida_.Forward(a0);
  SiluForward(pre_salida_, act_salida_);
  return conv_salida_.Forward(act_salida_);
}

Tensor UNet2D::Backward(const Tensor& dout) {
  Tensor d = conv_salida_.Backward(dout);
  Tensor d_pre;
  SiluBackward(d, pre_salida_, d_pre);
  d = norm_salida_.Backward(d_pre);                 // [N, C, H, W]

  // Deshacer la subida. `Concat` se deshace CORTANDO: la primera mitad de los
  // canales vuelve por la rama que subia y la segunda por el salto. Cortar en el
  // sitio equivocado mezclaria los dos gradientes y la red seguiria entrenando,
  // peor, sin que nada fallara.
  Tensor d_c0 = res_alta0_.Backward(d);             // [N, 2C, H, W]
  auto partir = [&](const Tensor& g, int canales_primera) {
    const int C = g.Shape()[1], H = g.Shape()[2], W = g.Shape()[3];
    const size_t espacial = static_cast<size_t>(H) * W;
    Tensor a({g.Shape()[0], canales_primera, H, W});
    Tensor b({g.Shape()[0], C - canales_primera, H, W});
    for (int n = 0; n < g.Shape()[0]; ++n) {
      const size_t org = static_cast<size_t>(n) * C * espacial;
      std::memcpy(a.Data() + static_cast<size_t>(n) * canales_primera * espacial,
                  g.Data() + org, canales_primera * espacial * sizeof(float));
      std::memcpy(b.Data() + static_cast<size_t>(n) * (C - canales_primera) * espacial,
                  g.Data() + org + canales_primera * espacial,
                  (C - canales_primera) * espacial * sizeof(float));
    }
    return std::make_pair(a, b);
  };

  auto [d_u0, d_salto0_arriba] = partir(d_c0, canales_);
  Tensor d_a1 = subir0_.Backward(d_u0);             // [N, C, H/2, W/2]

  Tensor d_c1 = res_alta1_.Backward(d_a1);          // [N, 4C, H/2, W/2]
  auto [d_u1, d_salto1_arriba] = partir(d_c1, 2 * canales_);
  Tensor d_centro = subir1_.Backward(d_u1);         // [N, 2C, H/4, W/4]

  Tensor d_d1 = res_centro_.Backward(d_centro);

  // El salto llega por DOS caminos: el de la bajada y el que viene de la
  // concatenacion. Se suman. Quedarse con uno es el mismo error que olvidar el
  // atajo de un bloque residual, pero mas dificil de ver.
  Tensor d_salto1 = bajar1_.Backward(d_d1);
  for (size_t i = 0; i < d_salto1.TotalSize(); ++i) d_salto1[i] += d_salto1_arriba[i];

  Tensor d_d0 = res_baja1_.Backward(d_salto1);
  Tensor d_salto0 = bajar0_.Backward(d_d0);
  for (size_t i = 0; i < d_salto0.TotalSize(); ++i) d_salto0[i] += d_salto0_arriba[i];

  Tensor d_h = res_baja0_.Backward(d_salto0);
  return conv_entrada_.Backward(d_h);
}

std::vector<Tensor*> UNet2D::GetParameters() {
  std::vector<Tensor*> out;
  for (Parameter* p : Parameters()) out.push_back(&p->Value());
  return out;
}

std::vector<Tensor*> UNet2D::GetGradients() {
  std::vector<Tensor*> out;
  for (Parameter* p : Parameters()) out.push_back(&p->Grad());
  return out;
}

bool UNet2D::GuardarPesos(const std::string& ruta) {
  const auto r = nsf::Save(ruta, nsf::FromNamedParameters(NamedParameters()),
                           MetadatosArquitectura());
  if (!r) std::cerr << "UNet2D: error al guardar: " << r.error << "\n";
  return r.ok;
}

bool UNet2D::CargarPesos(const std::string& ruta) {
  const auto r = nsf::Load(ruta, nsf::FromNamedParameters(NamedParameters()),
                           MetadatosArquitectura());
  if (!r) std::cerr << "UNet2D: error al cargar: " << r.error << "\n";
  return r.ok;
}

std::map<std::string, std::string> UNet2D::MetadatosArquitectura() const {
  return {{"arch", "unet2d"},
          {"canales_imagen", std::to_string(canales_imagen_)},
          {"canales", std::to_string(canales_)},
          {"dim_t", std::to_string(dim_t_)},
          {"grupos", std::to_string(grupos_)}};
}

size_t UNet2D::NumParametros() {
  size_t total = 0;
  for (const Tensor* p : GetParameters()) total += p->TotalSize();
  return total;
}

}  // namespace diffusion
}  // namespace neuralsuite
