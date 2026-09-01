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
#include "serialization.h"
#include "artifacts.h"
#include "gpt.h"
#include "image.h"
#include "layer.h"
#include "losses.h"
#include "optimizers.h"
#include "tensor.h"
#include "tokenizer.h"

// Specialized Neural Network Layers
#include "layers/attention.h"
#include "layers/conv2d.h"
#include "layers/embedding.h"
#include "layers/embedding_autograd.h"
#include "layers/graph_conv.h"
#include "layers/linear.h"
#include "layers/linear_autograd.h"
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
    for (Parameter* p : layer->Parameters()) params_.push_back(p);
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

  [[nodiscard]] std::vector<Parameter*>& Parameters() { return params_; }

  [[nodiscard]] std::vector<Tensor*> GetParameters() {
    std::vector<Tensor*> out;
    for (Parameter* p : params_) out.push_back(&p->Value());
    return out;
  }

  [[nodiscard]] std::vector<Tensor*> GetGradients() {
    std::vector<Tensor*> out;
    for (Parameter* p : params_) out.push_back(&p->Grad());
    return out;
  }

  bool Save(const std::string& filepath) {
    const auto result = nsf::Save(filepath, NamedTensors(),
                                 {{"arch", "sequential"},
                                  {"n_params", std::to_string(params_.size())}});
    if (!result) std::cerr << "Error al guardar: " << result.error << "\n";
    return result.ok;
  }

  bool Load(const std::string& filepath) {
    const auto result = nsf::Load(filepath, NamedTensors(),
                                 {{"arch", "sequential"},
                                  {"n_params", std::to_string(params_.size())}});
    if (!result) std::cerr << "Error al cargar: " << result.error << "\n";
    return result.ok;
  }

 private:
  /** @brief Nombra los parametros por su posicion en la secuencia de capas. */
  [[nodiscard]] std::vector<nsf::NamedTensor> NamedTensors() {
    std::vector<nsf::NamedTensor> out;
    for (size_t i = 0; i < params_.size(); ++i) {
      out.push_back({"param." + std::to_string(i), &params_[i]->Value()});
    }
    return out;
  }

 public:

 private:
  std::vector<std::shared_ptr<Layer>> layers_;
  std::vector<Parameter*> params_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_NEURALSUITE_H_
