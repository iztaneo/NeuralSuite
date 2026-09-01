// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/linear_autograd.h.

#include "layers/linear_autograd.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace neuralsuite {

Tensor LinearAutograd::Forward(const Tensor& input) {
  using autograd::Variable;

  forma_entrada_ = input.Shape();
  const int rango = static_cast<int>(forma_entrada_.size());
  const int in_dim = forma_entrada_[rango - 1];
  if (in_dim != in_features_) {
    throw std::invalid_argument(
        "LinearAutograd: la entrada termina en " + std::to_string(in_dim) +
        " y la capa espera " + std::to_string(in_features_) + ".");
  }
  // `Weight()` entrega una referencia mutable, asi que la tabla puede haber
  // sido sustituida por una de otra forma sin que nadie se entere hasta que el
  // producto da un resultado equivocado.
  if (weight_.Value().Shape() != std::vector<int>({in_features_, out_features_})) {
    throw std::invalid_argument("LinearAutograd: los pesos no son [entradas, salidas].");
  }
  const int n = static_cast<int>(input.TotalSize()) / in_dim;

  // Los parametros cambian tras cada paso del optimizador, asi que las hojas
  // del grafo se crean de nuevo en cada pasada a partir del valor vigente.
  entrada_ = Variable::Create(input.View({n, in_dim}), /*requires_grad=*/true);
  peso_ = Variable::Create(weight_.Value(), /*requires_grad=*/true);
  sesgo_ = Variable::Create(bias_.Value(), /*requires_grad=*/true);

  // Aqui esta toda la capa. El gradiente sale de estas dos lineas.
  salida_ = autograd::Add(autograd::MatMulVar(entrada_, peso_), sesgo_);

  std::vector<int> forma_salida = forma_entrada_;
  forma_salida[rango - 1] = out_features_;
  Tensor out = salida_->Value();
  out.Reshape(forma_salida);
  return out;
}

Tensor LinearAutograd::Backward(const Tensor& dout) {
  using autograd::Variable;

  const int n = static_cast<int>(dout.TotalSize()) / out_features_;

  // `Backward` exige una raiz escalar y siembra el gradiente el mismo. Para
  // propagar un `dout` concreto se cierra el grafo con una perdida cuya
  // derivada respecto a la salida es exactamente ese `dout`: sum(salida * dout).
  const Tensor dout_2d = dout.View({n, out_features_});
  autograd::Backward(
      autograd::Sum(autograd::Mul(salida_, Variable::Create(dout_2d))));

  // `Linear` asigna sus gradientes en vez de acumularlos; se copia igual para
  // que las dos capas sean intercambiables ante el optimizador.
  std::memcpy(weight_.Grad().Data(), peso_->Grad().Data(),
              weight_.Grad().TotalSize() * sizeof(float));
  std::memcpy(bias_.Grad().Data(), sesgo_->Grad().Data(),
              bias_.Grad().TotalSize() * sizeof(float));

  Tensor dx = entrada_->Grad();
  dx.Reshape(forma_entrada_);
  return dx;
}

}  // namespace neuralsuite
