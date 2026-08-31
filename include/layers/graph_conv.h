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
    Register(&linear_, "linear");
  }

  /**
   * @brief Forward pass of GCN layer: H_out = ReLU(A_norm * H_in * W)
   * @param input Node features matrix H_in [Num_Nodes, In_Features]
   * @param adj_norm Normalized adjacency matrix A_norm [Num_Nodes, Num_Nodes]
   */
  Tensor ForwardWithAdj(const Tensor& input, const Tensor& adj_norm);

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

 private:
  Linear linear_;
  Activation relu_;
  Tensor last_input_;
  Tensor last_adj_norm_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_GRAPH_CONV_H_
