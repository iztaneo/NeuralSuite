// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/graph_conv.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/graph_conv.h"

namespace neuralsuite {

Tensor GraphConv::ForwardWithAdj(const Tensor& input, const Tensor& adj_norm) {
    last_input_ = input;
    last_adj_norm_ = adj_norm;

    // 1. Transformación de características por nodo: H_linear = H_in * W
    Tensor H_linear = linear_.Forward(input);

    // 2. Agregación de vecinos por grafo: H_agg = A_norm * H_linear
    int num_nodes = input.Shape()[0];
    int out_feat = H_linear.Shape()[1];

    Tensor H_agg({num_nodes, out_feat});
    MatMul(adj_norm, H_linear, H_agg);

    // 3. Activación no lineal
    return relu_.Forward(H_agg);
  }

Tensor GraphConv::Forward(const Tensor& input) {
    // Implementación estándar de capa usando matriz identidad como grafo por defecto
    int num_nodes = input.Shape()[0];
    Tensor eye({num_nodes, num_nodes});
    eye.Zeros();
    for (int i = 0; i < num_nodes; ++i) eye[i * num_nodes + i] = 1.0f;
    return ForwardWithAdj(input, eye);
  }

Tensor GraphConv::Backward(const Tensor& dout) {
    Tensor dH_agg = relu_.Backward(dout);

    // Backward de agregación de grafo: dH_linear = A_norm^T * dH_agg
    Tensor adj_T = Transpose(last_adj_norm_);
    Tensor dH_linear({dH_agg.Shape()[0], dH_agg.Shape()[1]});
    MatMul(adj_T, dH_agg, dH_linear);

    return linear_.Backward(dH_linear);
  }

}  // namespace neuralsuite
