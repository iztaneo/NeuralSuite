// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file neuralsuite.h
 * @brief Master Header for the NeuralSuite C++ Neural Network & LLM Library.
 * 
 * Including this single header grants access to the full NeuralSuite framework API:
 * Tensors, Layers (Linear, Conv2D, LSTM, MultiHeadAttention, Embedding), Activations,
 * Loss Functions, Optimizers (AdamW), Models (GPTModel, Sequential), and Tokenizers.
 */

#ifndef NEURAL_SUITE_INCLUDE_NEURALSUITE_H_
#define NEURAL_SUITE_INCLUDE_NEURALSUITE_H_

#include <memory>
#include <string>
#include <vector>

// Core Primitives & Base Classes
#include "activations.h"
#include "artifacts.h"
#include "gpt.h"
#include "layer.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"
#include "tokenizer.h"

// Specialized Neural Network Layers
#include "layers/attention.h"
#include "layers/conv2d.h"
#include "layers/embedding.h"
#include "layers/graph_conv.h"
#include "layers/linear.h"
#include "layers/lstm.h"
#include "layers/maxpool2d.h"
#include "layers/residual.h"
// Reusable Neural Network Models
#include "models/ocr.h"
namespace neuralsuite {

/**
 * @class Sequential
 * @brief High-level container for stacking neural network layers in sequence (Keras/PyTorch style).
 */
class Sequential {
 public:
  Sequential() = default;

  void Add(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
    for (auto p : layer->GetParameters()) params_.push_back(p);
    for (auto g : layer->GetGradients()) grads_.push_back(g);
  }

  Tensor Forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers_) {
      x = layer->Forward(x);
    }
    return x;
  }

  Tensor Backward(const Tensor& dout) {
    Tensor dx = dout;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      dx = (*it)->Backward(dx);
    }
    return dx;
  }

  std::vector<Tensor*>& GetParameters() { return params_; }
  std::vector<Tensor*>& GetGradients() { return grads_; }

  bool Save(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    for (auto p : params_) {
      out.write(reinterpret_cast<const char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

  bool Load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;
    for (auto p : params_) {
      in.read(reinterpret_cast<char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<Layer>> layers_;
  std::vector<Tensor*> params_;
  std::vector<Tensor*> grads_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_NEURALSUITE_H_
