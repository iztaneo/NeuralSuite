// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file conv2d.h
 * @brief 2D Convolutional Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_

#include <vector>
#include "../layer.h"

namespace neuralsuite {

/**
 * @class Conv2D
 * @brief 2D Convolution Layer for Image Processing.
 */
class Conv2D : public Layer {
 public:
  Conv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
      : in_channels_(in_ch),
        out_channels_(out_ch),
        kernel_size_(k_size),
        stride_(str),
        padding_(pad),
        weight_({out_ch, in_ch, k_size, k_size}),
        bias_({out_ch}),
        dweight_({out_ch, in_ch, k_size, k_size}),
        dbias_({out_ch}) {
    weight_.XavierInit(in_ch * k_size * k_size, out_ch * k_size * k_size);
    bias_.Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int height = input.Shape()[2];
    int width = input.Shape()[3];

    int out_h = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (width + 2 * padding_ - kernel_size_) / stride_ + 1;

    Tensor output({batch_size, out_channels_, out_h, out_w});

    for (int b = 0; b < batch_size; ++b) {
      for (int oc = 0; oc < out_channels_; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            float val = bias_[oc];
            for (int ic = 0; ic < in_channels_; ++ic) {
              for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                  int ih = oh * stride_ + kh - padding_;
                  int iw = ow * stride_ + kw - padding_;
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    size_t in_idx = ((b * in_channels_ + ic) * height + ih) * width + iw;
                    size_t w_idx = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;
                    val += input[in_idx] * weight_[w_idx];
                  }
                }
              }
            }
            size_t out_idx = ((b * out_channels_ + oc) * out_h + oh) * out_w + ow;
            output[out_idx] = val;
          }
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    int batch_size = last_input_.Shape()[0];
    int height = last_input_.Shape()[2];
    int width = last_input_.Shape()[3];

    int out_h = dout.Shape()[2];
    int out_w = dout.Shape()[3];

    Tensor dx({batch_size, in_channels_, height, width});
    dx.Zeros();
    dweight_.Zeros();
    dbias_.Zeros();

    for (int b = 0; b < batch_size; ++b) {
      for (int oc = 0; oc < out_channels_; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            size_t dout_idx = ((b * out_channels_ + oc) * out_h + oh) * out_w + ow;
            float d = dout[dout_idx];
            dbias_[oc] += d;

            for (int ic = 0; ic < in_channels_; ++ic) {
              for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                  int ih = oh * stride_ + kh - padding_;
                  int iw = ow * stride_ + kw - padding_;
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    size_t in_idx = ((b * in_channels_ + ic) * height + ih) * width + iw;
                    size_t w_idx = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;

                    dweight_[w_idx] += d * last_input_[in_idx];
                    dx[in_idx] += d * weight_[w_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
    return dx;
  }

  std::vector<Tensor*> GetParameters() override { return {&weight_, &bias_}; }
  std::vector<Tensor*> GetGradients() override { return {&dweight_, &dbias_}; }

 private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  Tensor weight_;
  Tensor bias_;
  Tensor dweight_;
  Tensor dbias_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
