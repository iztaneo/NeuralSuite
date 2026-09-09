// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/swiglu.h.

#include "layers/swiglu.h"

namespace neuralsuite {

Tensor SwiGLU::Forward(const Tensor& input) {
  g_lineal_ = w_gate_.Forward(input);
  u_ = w_up_.Forward(input);

  // La activacion va en la PUERTA, no en la proyeccion de arriba. Al reves
  // tambien da numeros razonables y la red entrena algo peor, que es lo que
  // hace el error dificil de ver.
  SiluForward(g_lineal_, g_act_);

  Tensor producto(g_act_.Shape());
  for (size_t i = 0; i < producto.TotalSize(); ++i) producto[i] = g_act_[i] * u_[i];

  return w_down_.Forward(producto);
}

Tensor SwiGLU::Backward(const Tensor& dout) {
  const Tensor dproducto = w_down_.Backward(dout);

  // La regla del producto reparte el gradiente entre las dos ramas, cada una
  // multiplicada por el valor de la otra.
  Tensor d_gact(g_act_.Shape()), du(u_.Shape());
  for (size_t i = 0; i < dproducto.TotalSize(); ++i) {
    d_gact[i] = dproducto[i] * u_[i];
    du[i] = dproducto[i] * g_act_[i];
  }

  // SiluBackward pide la ENTRADA de la activacion, no su salida: SiLU no es
  // inyectiva y no se puede recuperar x de y.
  Tensor d_glineal;
  SiluBackward(d_gact, g_lineal_, d_glineal);

  const Tensor dx_gate = w_gate_.Backward(d_glineal);
  const Tensor dx_up = w_up_.Backward(du);

  // La entrada alimenta las dos proyecciones, asi que recibe la SUMA. Es el
  // mismo reparto que en CrossAttention con K y V.
  Tensor dx = dx_gate;
  for (size_t i = 0; i < dx.TotalSize(); ++i) dx[i] += dx_up[i];
  return dx;
}

}  // namespace neuralsuite
