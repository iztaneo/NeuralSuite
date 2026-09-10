// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de diffusion/resblock.h.

#include "diffusion/resblock.h"

#include <cstring>

namespace neuralsuite {
namespace diffusion {

ResBlockTiempo::ResBlockTiempo(int canales_entrada, int canales_salida, int dim_tiempo,
                               int grupos)
    : c_out_(canales_salida),
      dim_t_(dim_tiempo),
      norm1_(grupos, canales_entrada),
      norm2_(grupos, canales_salida),
      conv1_(canales_entrada, canales_salida, 3, 1, 1),
      conv2_(canales_salida, canales_salida, 3, 1, 1),
      proy_t_(dim_tiempo, canales_salida) {
  Register(&norm1_, "norm1");
  Register(&conv1_, "conv1");
  Register(&proy_t_, "proy_t");
  Register(&norm2_, "norm2");
  Register(&conv2_, "conv2");

  // El atajo solo lleva convolucion cuando hay que cambiar de canales. Si no,
  // es la identidad, y es asi como el gradiente llega intacto hacia abajo.
  if (canales_entrada != canales_salida) {
    atajo_ = std::make_unique<Conv2D>(canales_entrada, canales_salida, 1, 1, 0);
    Register(atajo_.get(), "atajo");
  }
}

Tensor ResBlockTiempo::Forward(const Tensor& x, const Tensor& t_emb) {
  if (x.Shape().size() != 4) {
    throw std::invalid_argument("ResBlockTiempo: x debe ser [N, C, alto, ancho].");
  }
  if (t_emb.Shape().size() != 2 || t_emb.Shape()[1] != dim_t_) {
    throw std::invalid_argument(
        "ResBlockTiempo: el embedding del paso debe ser [N, " +
        std::to_string(dim_t_) + "].");
  }
  if (t_emb.Shape()[0] != x.Shape()[0]) {
    throw std::invalid_argument("ResBlockTiempo: x y el paso no comparten lote.");
  }

  x_ = x;
  n_ = x.Shape()[0];
  h_ = x.Shape()[2];
  w_ = x.Shape()[3];

  // Rama principal, primera mitad.
  h1_pre_ = norm1_.Forward(x);
  SiluForward(h1_pre_, h1_act_);
  Tensor h = conv1_.Forward(h1_act_);          // [N, Cout, H, W]

  // El tiempo entra aqui, sumado POR CANAL: un valor por canal que se aplica a
  // todo su mapa. El paso es una propiedad de la imagen entera.
  t_pre_ = t_emb;
  SiluForward(t_pre_, t_act_);
  const Tensor t_proy = proy_t_.Forward(t_act_);   // [N, Cout]

  const size_t espacial = static_cast<size_t>(h_) * w_;
  for (int n = 0; n < n_; ++n) {
    for (int c = 0; c < c_out_; ++c) {
      const float valor = t_proy[static_cast<size_t>(n) * c_out_ + c];
      const size_t base = (static_cast<size_t>(n) * c_out_ + c) * espacial;
      for (size_t s = 0; s < espacial; ++s) h[base + s] += valor;
    }
  }
  h2_ = h;

  // Segunda mitad.
  h2_pre_ = norm2_.Forward(h2_);
  SiluForward(h2_pre_, h2_act_);
  Tensor salida = conv2_.Forward(h2_act_);

  const Tensor atajo = atajo_ ? atajo_->Forward(x) : x;
  for (size_t i = 0; i < salida.TotalSize(); ++i) salida[i] += atajo[i];
  return salida;
}

Tensor ResBlockTiempo::Backward(const Tensor& dout) {
  // El residuo reparte el gradiente por igual a las dos ramas: la principal y
  // el atajo. Olvidar una de las dos es el error clasico de un bloque residual,
  // y deja la red entrenando peor sin que nada falle.
  const Tensor d_conv2 = conv2_.Backward(dout);

  Tensor d_h2_pre;
  SiluBackward(d_conv2, h2_pre_, d_h2_pre);
  Tensor d_h2 = norm2_.Backward(d_h2_pre);      // [N, Cout, H, W]

  // El tiempo se sumo a todo el mapa, asi que hacia atras recibe la SUMA sobre
  // las posiciones espaciales. Es el adjunto de repartir un valor por canal.
  const size_t espacial = static_cast<size_t>(h_) * w_;
  Tensor d_t_proy({n_, c_out_});
  for (int n = 0; n < n_; ++n) {
    for (int c = 0; c < c_out_; ++c) {
      const size_t base = (static_cast<size_t>(n) * c_out_ + c) * espacial;
      double acc = 0.0;
      for (size_t s = 0; s < espacial; ++s) acc += d_h2[base + s];
      d_t_proy[static_cast<size_t>(n) * c_out_ + c] = static_cast<float>(acc);
    }
  }
  const Tensor d_t_act = proy_t_.Backward(d_t_proy);
  SiluBackward(d_t_act, t_pre_, d_t_emb_);

  // `d_h2` sigue siendo el gradiente de la salida de conv1: sumar el tiempo no
  // cambia lo que llega a la convolucion, solo anade una rama.
  const Tensor d_h1_act = conv1_.Backward(d_h2);
  Tensor d_h1_pre;
  SiluBackward(d_h1_act, h1_pre_, d_h1_pre);
  Tensor dx = norm1_.Backward(d_h1_pre);

  const Tensor d_atajo = atajo_ ? atajo_->Backward(dout) : dout;
  for (size_t i = 0; i < dx.TotalSize(); ++i) dx[i] += d_atajo[i];
  return dx;
}

}  // namespace diffusion
}  // namespace neuralsuite
