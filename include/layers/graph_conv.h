// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file graph_conv.h
 * @brief Graph Convolutional Layer (GCN - Kipf & Welling, 2017) following Google C++ Style Guide.
 * 
 * Implements Graph Convolution: H_next = ReLU(A_hat * H * W)
 * where A_hat is the normalized adjacency matrix with self-loops.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_GRAPH_CONV_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_GRAPH_CONV_H_

#include <cmath>
#include <memory>
#include <vector>
#include "../activations.h"
#include "../layer.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class GraphConv
 * @brief Graph Convolutional Layer for Graph Neural Networks (GNN / GCN).
 */
class GraphConv : public Layer {
 public:
  GraphConv(int in_features, int out_features)
      : linear_(in_features, out_features), relu_(ActivationType::kRelu) {
    Register(&linear_);
  }

  /**
   * @brief Forward pass of GCN layer: H_out = ReLU(A_norm * H_in * W)
   * @param input Node features matrix H_in [Num_Nodes, In_Features]
   * @param adj_norm Normalized adjacency matrix A_norm [Num_Nodes, Num_Nodes]
   */
  Tensor ForwardWithAdj(const Tensor& input, const Tensor& adj_norm) {
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

  Tensor Forward(const Tensor& input) override {
    // Implementación estándar de capa usando matriz identidad como grafo por defecto
    int num_nodes = input.Shape()[0];
    Tensor eye({num_nodes, num_nodes});
    eye.Zeros();
    for (int i = 0; i < num_nodes; ++i) eye[i * num_nodes + i] = 1.0f;
    return ForwardWithAdj(input, eye);
  }

  Tensor Backward(const Tensor& dout) override {
    Tensor dH_agg = relu_.Backward(dout);

    // Backward de agregación de grafo: dH_linear = A_norm^T * dH_agg
    Tensor adj_T = Transpose(last_adj_norm_);
    Tensor dH_linear({dH_agg.Shape()[0], dH_agg.Shape()[1]});
    MatMul(adj_T, dH_agg, dH_linear);

    return linear_.Backward(dH_linear);
  }

 private:
  Linear linear_;
  Activation relu_;
  Tensor last_input_;
  Tensor last_adj_norm_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_GRAPH_CONV_H_
