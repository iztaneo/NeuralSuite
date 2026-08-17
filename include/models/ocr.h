// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr.h
 * @brief Full CRNN (Convolutional Recurrent Neural Network) OCR Library Model.
 * 
 * Reusable OCR Model combining CNN Feature Extractor (Conv2D + MaxPool2D)
 * with Sequence Classification & Character Tokenizer Decoding.
 */

#ifndef NEURAL_SUITE_INCLUDE_MODELS_OCR_H_
#define NEURAL_SUITE_INCLUDE_MODELS_OCR_H_

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "../activations.h"
#include "../layer.h"
#include "../layers/conv2d.h"
#include "../layers/linear.h"
#include "../layers/maxpool2d.h"
#include "../tensor.h"
#include "../tokenizer.h"

namespace neuralsuite {

/**
 * @class CRNNModel
 * @brief High-level reusable OCR Model for image-to-text recognition.
 */
class CRNNModel : public Layer {
 public:
  CRNNModel(int in_channels, int hidden_dim, int num_classes)
      : conv_(in_channels, 4, 3, 1, 0),
        relu_(ActivationType::kRelu),
        pool_(2, 2),
        fc1_(4 * 3 * 3, hidden_dim),
        fc2_(hidden_dim, num_classes),
        num_classes_(num_classes) {}

  Tensor Forward(const Tensor& input_images) override {
    last_input_ = input_images;
    int batch_size = input_images.Shape()[0];

    // 1. Extractor de Características Visuales CNN 2D
    Tensor h_conv = conv_.Forward(input_images);
    Tensor h_relu = relu_.Forward(h_conv);
    Tensor h_pool = pool_.Forward(h_relu);

    // 2. Aplanar mapas de características
    last_h_flat_ = Tensor({batch_size, 4 * 3 * 3});
    std::memcpy(last_h_flat_.Data(), h_pool.Data(), h_pool.TotalSize() * sizeof(float));

    // 3. Proyección Secuencial a Vocabulario de Caracteres
    Tensor h_fc1 = fc1_.Forward(last_h_flat_);
    Tensor a_fc1 = relu_.Forward(h_fc1);
    Tensor logits = fc2_.Forward(a_fc1);

    return logits;
  }

  Tensor Backward(const Tensor& dout) override {
    int batch_size = last_input_.Shape()[0];

    // 1. Backward de capas densas de clasificación
    Tensor da_fc1 = fc2_.Backward(dout);
    Tensor dh_fc1 = relu_.Backward(da_fc1);
    Tensor dh_flat = fc1_.Backward(dh_fc1);

    // 2. Backward de extractor convolucional visual 2D
    Tensor dh_pool({batch_size, 4, 3, 3});
    std::memcpy(dh_pool.Data(), dh_flat.Data(), dh_flat.TotalSize() * sizeof(float));

    Tensor dh_relu = pool_.Backward(dh_pool);
    Tensor dh_conv = relu_.Backward(dh_relu);
    Tensor dinput = conv_.Backward(dh_conv);

    return dinput;
  }

  std::string Decode(const Tensor& logits, const std::vector<char>& vocab) {
    int batch_size = logits.Shape()[0];
    int num_classes = logits.Shape()[1];
    std::string result = "";

    for (int b = 0; b < batch_size; ++b) {
      int best_class = 0;
      float max_val = logits[b * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        if (logits[b * num_classes + c] > max_val) {
          max_val = logits[b * num_classes + c];
          best_class = c;
        }
      }
      if (best_class < static_cast<int>(vocab.size())) {
        result += vocab[best_class];
      }
    }
    return result;
  }

  std::vector<Tensor*> GetParameters() override {
    std::vector<Tensor*> p = conv_.GetParameters();
    auto p1 = fc1_.GetParameters();
    auto p2 = fc2_.GetParameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
  }

  std::vector<Tensor*> GetGradients() override {
    std::vector<Tensor*> g = conv_.GetGradients();
    auto g1 = fc1_.GetGradients();
    auto g2 = fc2_.GetGradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    return g;
  }

  bool Save(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    for (auto p : GetParameters()) {
      out.write(reinterpret_cast<const char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

  bool Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    for (auto p : GetParameters()) {
      in.read(reinterpret_cast<char*>(p->Data()), p->TotalSize() * sizeof(float));
    }
    return true;
  }

 private:
  Conv2D conv_;
  Activation relu_;
  MaxPool2D pool_;
  Linear fc1_;
  Linear fc2_;
  int num_classes_;

  Tensor last_input_;
  Tensor last_h_flat_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_MODELS_OCR_H_
