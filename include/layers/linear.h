/**
 * @file linear.h
 * @brief Capa Densa (Fully Connected / Linear) Y = X*W + b
 */

#ifndef NEURAL_SUITE_LINEAR_H
#define NEURAL_SUITE_LINEAR_H

#include "../layer.h"

namespace ns {

class Linear : public Layer {
public:
    int in_features;
    int out_features;
    Tensor weight; // Shape: [in_features, out_features]
    Tensor bias;   // Shape: [out_features]
    Tensor dweight;
    Tensor dbias;
    Tensor last_input;

    Linear(int in_dim, int out_dim)
        : in_features(in_dim), out_features(out_dim),
          weight({in_dim, out_dim}), bias({out_dim}),
          dweight({in_dim, out_dim}), dbias({out_dim}) {
        weight.xavier_init(in_dim, out_dim);
        bias.zeros();
    }

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int B = input.shape[0];
        Tensor output({B, out_features});
        
        matmul(input, weight, output);

        // Sumar bias
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < out_features; ++j) {
                output.data[i * out_features + j] += bias.data[j];
            }
        }
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        int B = last_input.shape[0];
        Tensor dx({B, in_features});

        // dx = dout * W^T
        Tensor W_T = transpose(weight);
        matmul(dout, W_T, dx);

        // dW = X^T * dout
        Tensor X_T = transpose(last_input);
        matmul(X_T, dout, dweight);

        // db = sum_batch(dout)
        dbias.zeros();
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < out_features; ++j) {
                dbias.data[j] += dout.data[i * out_features + j];
            }
        }
        return dx;
    }

    std::vector<Tensor*> get_parameters() override { return {&weight, &bias}; }
    std::vector<Tensor*> get_gradients() override { return {&dweight, &dbias}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LINEAR_H
