/**
 * @file maxpool2d.h
 * @brief Capa de Max Pooling 2D para redes CNNs.
 */

#ifndef NEURAL_SUITE_MAXPOOL2D_H
#define NEURAL_SUITE_MAXPOOL2D_H

#include "../layer.h"

namespace ns {

class MaxPool2D : public Layer {
public:
    int pool_size;
    int stride;
    Tensor last_input;
    std::vector<size_t> max_indices;

    MaxPool2D(int p_size = 2, int str = 2) : pool_size(p_size), stride(str) {}

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int B = input.shape[0];
        int C = input.shape[1];
        int H = input.shape[2];
        int W = input.shape[3];

        int out_h = (H - pool_size) / stride + 1;
        int out_w = (W - pool_size) / stride + 1;

        Tensor output({B, C, out_h, out_w});
        max_indices.resize(output.total_size());

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float max_val = -1e9f;
                        size_t max_idx = 0;

                        for (int ph = 0; ph < pool_size; ++ph) {
                            for (int pw = 0; pw < pool_size; ++pw) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                size_t in_idx = ((b * C + c) * H + ih) * W + iw;
                                if (input.data[in_idx] > max_val) {
                                    max_val = input.data[in_idx];
                                    max_idx = in_idx;
                                }
                            }
                        }

                        size_t out_idx = ((b * C + c) * out_h + oh) * out_w + ow;
                        output.data[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        Tensor dx(last_input.shape);
        dx.zeros();

        size_t out_sz = dout.total_size();
        for (size_t i = 0; i < out_sz; ++i) {
            size_t max_idx = max_indices[i];
            dx.data[max_idx] += dout.data[i];
        }
        return dx;
    }
};

} // namespace ns

#endif // NEURAL_SUITE_MAXPOOL2D_H
