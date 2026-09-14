// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/resblock2d.h.

#include "layers/resblock2d.h"

#include <stdexcept>

namespace neuralsuite {

ResBlock2D::ResBlock2D(int canales_entrada, int canales_salida, int grupos)
    : norm1_(grupos, canales_entrada),
      norm2_(grupos, canales_salida),
      conv1_(canales_entrada, canales_salida, 3, 1, 1),
      conv2_(canales_salida, canales_salida, 3, 1, 1) {
  Register(&norm1_, "norm1");
  Register(&conv1_, "conv1");
  Register(&norm2_, "norm2");
  Register(&conv2_, "conv2");
  if (canales_entrada != canales_salida) {
    atajo_ = std::make_unique<Conv2D>(canales_entrada, canales_salida, 1, 1, 0);
    Register(atajo_.get(), "atajo");
  }
}

Tensor ResBlock2D::Forward(const Tensor& x) {
  if (x.Shape().size() != 4) {
    throw std::invalid_argument("ResBlock2D: x debe ser [N, C, alto, ancho].");
  }
  h1_pre_ = norm1_.Forward(x);
  SiluForward(h1_pre_, h1_act_);
  const Tensor h = conv1_.Forward(h1_act_);

  h2_pre_ = norm2_.Forward(h);
  SiluForward(h2_pre_, h2_act_);
  Tensor salida = conv2_.Forward(h2_act_);

  const Tensor atajo = atajo_ ? atajo_->Forward(x) : x;
  for (size_t i = 0; i < salida.TotalSize(); ++i) salida[i] += atajo[i];
  return salida;
}

Tensor ResBlock2D::Backward(const Tensor& dout) {
  // El residuo reparte el gradiente a las dos ramas. Olvidar el atajo es el
  // error clasico: la red sigue entrenando, peor, y nada falla.
  const Tensor d_conv2 = conv2_.Backward(dout);
  Tensor d_h2_pre;
  SiluBackward(d_conv2, h2_pre_, d_h2_pre);
  const Tensor d_h = norm2_.Backward(d_h2_pre);

  const Tensor d_h1_act = conv1_.Backward(d_h);
  Tensor d_h1_pre;
  SiluBackward(d_h1_act, h1_pre_, d_h1_pre);
  Tensor dx = norm1_.Backward(d_h1_pre);

  const Tensor d_atajo = atajo_ ? atajo_->Backward(dout) : dout;
  for (size_t i = 0; i < dx.TotalSize(); ++i) dx[i] += d_atajo[i];
  return dx;
}

}  // namespace neuralsuite
