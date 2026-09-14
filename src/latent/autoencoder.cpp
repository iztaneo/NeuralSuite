// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de latent/autoencoder.h.

#include "latent/autoencoder.h"

#include <stdexcept>
#include <string>

namespace neuralsuite {
namespace latent {

Codificador::Codificador(int canales_imagen, int canales, int canales_latente, int grupos)
    : canales_imagen_(canales_imagen),
      conv_entrada_(canales_imagen, canales, 3, 1, 1),
      res0_(canales, canales, grupos),
      res1_(canales, 2 * canales, grupos),
      res2_(2 * canales, 2 * canales, grupos),
      bajar0_(2),
      bajar1_(2),
      norm_salida_(grupos, 2 * canales),
      conv_salida_(2 * canales, 2 * canales_latente, 3, 1, 1) {
  // El orden de registro es el de los nombres en el archivo de pesos. El
  // remuestreo no tiene parametros y no se registra.
  Register(&conv_entrada_, "conv_entrada");
  Register(&res0_, "res0");
  Register(&res1_, "res1");
  Register(&res2_, "res2");
  Register(&norm_salida_, "norm_salida");
  Register(&conv_salida_, "conv_salida");
}

Tensor Codificador::Forward(const Tensor& x) {
  if (x.Shape().size() != 4 || x.Shape()[1] != canales_imagen_) {
    throw std::invalid_argument("Codificador: la entrada debe ser [N, " +
                                std::to_string(canales_imagen_) + ", H, W].");
  }
  if (x.Shape()[2] % 4 != 0 || x.Shape()[3] % 4 != 0) {
    throw std::invalid_argument(
        "Codificador: alto y ancho deben ser multiplos de 4 (f = 4); llegaron " +
        std::to_string(x.Shape()[2]) + "x" + std::to_string(x.Shape()[3]) +
        ". MNIST se rellena a 32x32 por esto.");
  }
  Tensor h = conv_entrada_.Forward(x);
  h = res0_.Forward(h);
  h = bajar0_.Forward(h);
  h = res1_.Forward(h);
  h = bajar1_.Forward(h);
  h = res2_.Forward(h);
  pre_salida_ = norm_salida_.Forward(h);
  SiluForward(pre_salida_, act_salida_);
  return conv_salida_.Forward(act_salida_);
}

Tensor Codificador::Backward(const Tensor& dout) {
  const Tensor d_act = conv_salida_.Backward(dout);
  Tensor d_pre;
  SiluBackward(d_act, pre_salida_, d_pre);
  Tensor d = norm_salida_.Backward(d_pre);
  d = res2_.Backward(d);
  d = bajar1_.Backward(d);
  d = res1_.Backward(d);
  d = bajar0_.Backward(d);
  d = res0_.Backward(d);
  return conv_entrada_.Backward(d);
}

Decodificador::Decodificador(int canales_latente, int canales, int canales_imagen, int grupos)
    : canales_latente_(canales_latente),
      conv_entrada_(canales_latente, 2 * canales, 3, 1, 1),
      res0_(2 * canales, 2 * canales, grupos),
      res1_(2 * canales, canales, grupos),
      res2_(canales, canales, grupos),
      subir0_(2),
      subir1_(2),
      norm_salida_(grupos, canales),
      conv_salida_(canales, canales_imagen, 3, 1, 1) {
  Register(&conv_entrada_, "conv_entrada");
  Register(&res0_, "res0");
  Register(&res1_, "res1");
  Register(&res2_, "res2");
  Register(&norm_salida_, "norm_salida");
  Register(&conv_salida_, "conv_salida");
}

Tensor Decodificador::Forward(const Tensor& z) {
  if (z.Shape().size() != 4 || z.Shape()[1] != canales_latente_) {
    throw std::invalid_argument("Decodificador: el latente debe ser [N, " +
                                std::to_string(canales_latente_) + ", h, w].");
  }
  Tensor h = conv_entrada_.Forward(z);
  h = res0_.Forward(h);
  h = subir0_.Forward(h);
  h = res1_.Forward(h);
  h = subir1_.Forward(h);
  h = res2_.Forward(h);
  pre_salida_ = norm_salida_.Forward(h);
  SiluForward(pre_salida_, act_salida_);
  return conv_salida_.Forward(act_salida_);
}

Tensor Decodificador::Backward(const Tensor& dout) {
  const Tensor d_act = conv_salida_.Backward(dout);
  Tensor d_pre;
  SiluBackward(d_act, pre_salida_, d_pre);
  Tensor d = norm_salida_.Backward(d_pre);
  d = res2_.Backward(d);
  d = subir1_.Backward(d);
  d = res1_.Backward(d);
  d = subir0_.Backward(d);
  d = res0_.Backward(d);
  return conv_entrada_.Backward(d);
}

}  // namespace latent
}  // namespace neuralsuite
