// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file maxpool2d.h
 * @brief 2D Max Pooling Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class MaxPool2D
 * @brief 2D Max Pooling Layer for spatial downsampling.
 */
class MaxPool2D : public Layer {
 public:
  explicit MaxPool2D(int p_size = 2, int str = 2)
      : pool_size_(p_size), stride_(str) {}

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int channels = input.Shape()[1];
    int height = input.Shape()[2];
    int width = input.Shape()[3];

    int out_h = (height - pool_size_) / stride_ + 1;
    int out_w = (width - pool_size_) / stride_ + 1;

    Tensor output({batch_size, channels, out_h, out_w});
    max_indices_.resize(output.TotalSize());

    for (int b = 0; b < batch_size; ++b) {
      for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            float max_val = -1e9f;
            size_t max_idx = 0;

            for (int ph = 0; ph < pool_size_; ++ph) {
              for (int pw = 0; pw < pool_size_; ++pw) {
                int ih = oh * stride_ + ph;
                int iw = ow * stride_ + pw;
                size_t in_idx = ((b * channels + c) * height + ih) * width + iw;
                if (input[in_idx] > max_val) {
                  max_val = input[in_idx];
                  max_idx = in_idx;
                }
              }
            }

            size_t out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[out_idx] = max_val;
            max_indices_[out_idx] = max_idx;
          }
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    Tensor dx(last_input_.Shape());
    dx.Zeros();

    size_t out_sz = dout.TotalSize();
    for (size_t i = 0; i < out_sz; ++i) {
      size_t max_idx = max_indices_[i];
      dx[max_idx] += dout[i];
    }
    return dx;
  }

 private:
  int pool_size_;
  int stride_;
  Tensor last_input_;
  std::vector<size_t> max_indices_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_MAXPOOL2D_H_
