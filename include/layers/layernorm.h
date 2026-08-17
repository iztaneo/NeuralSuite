/**
 * @file layernorm.h
 * @brief Capa de Normalización LayerNorm para Transformers.
 */

#ifndef NEURAL_SUITE_LAYERNORM_H
#define NEURAL_SUITE_LAYERNORM_H

#include "../layer.h"

namespace ns {

class LayerNormLayer : public Layer {
public:
    int normalized_shape;
    float eps;
    Tensor gamma;
    Tensor beta;
    Tensor dgamma;
    Tensor dbeta;

    Tensor last_input;
    Tensor mean_cache;
    Tensor rstd_cache;

    LayerNormLayer(int shape, float epsilon = 1e-5f)
        : normalized_shape(shape), eps(epsilon),
          gamma({shape}), beta({shape}),
          dgamma({shape}), dbeta({shape}) {
        gamma.ones();
        beta.zeros();
    }

    Tensor forward(const Tensor& input) override {
        last_input = input;
        Tensor output(input.shape);
        layernorm_forward(input, gamma, beta, output, mean_cache, rstd_cache, eps);
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        Tensor dx(last_input.shape);
        layernorm_backward(dout, last_input, gamma, mean_cache, rstd_cache, dx, dgamma, dbeta);
        return dx;
    }

    std::vector<Tensor*> get_parameters() override { return {&gamma, &beta}; }
    std::vector<Tensor*> get_gradients() override { return {&dgamma, &dbeta}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LAYERNORM_H
