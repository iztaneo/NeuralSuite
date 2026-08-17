/**
 * @file optimizers.h
 * @brief Optimizadores de parámetros (AdamW y SGD con Momentum) en C++17 puro.
 */

#ifndef NEURAL_SUITE_OPTIMIZERS_H
#define NEURAL_SUITE_OPTIMIZERS_H

#include "tensor.h"
#include <vector>

namespace ns {

/**
 * @class Optimizer
 * @brief Interfaz abstracta para optimizadores de gradiente.
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /** @brief Actualiza los pesos de los tensores según sus gradientes */
    virtual void step() = 0;

    /** @brief Reinicia a cero todos los acumuladores de gradiente */
    virtual void zero_grad() = 0;
};

/**
 * @class SGD
 * @brief Descenso por el Gradiente Estocástico con Momentum.
 */
class SGD : public Optimizer {
public:
    std::vector<Tensor*> params;
    std::vector<Tensor*> grads;
    std::vector<Tensor> velocities;
    float lr;
    float momentum;

    SGD(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients, float learning_rate = 0.01f, float mom = 0.9f)
        : params(parameters), grads(gradients), lr(learning_rate), momentum(mom) {
        for (auto p : params) {
            Tensor v(p->shape);
            v.zeros();
            velocities.push_back(v);
        }
    }

    void step() override {
        for (size_t i = 0; i < params.size(); ++i) {
            size_t sz = params[i]->total_size();
            for (size_t k = 0; k < sz; ++k) {
                velocities[i].data[k] = momentum * velocities[i].data[k] + lr * grads[i]->data[k];
                params[i]->data[k] -= velocities[i].data[k];
            }
        }
    }

    void zero_grad() override {
        for (auto g : grads) g->zeros();
    }
};

/**
 * @class AdamW
 * @brief Optimizador Adaptive Moment Estimation con Weight Decay Desacoplado (Loshchilov & Hutter 2019).
 */
class AdamW : public Optimizer {
public:
    std::vector<Tensor*> params;
    std::vector<Tensor*> grads;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count;

    AdamW(const std::vector<Tensor*>& parameters, const std::vector<Tensor*>& gradients,
          float learning_rate = 1e-3f, float b1 = 0.9f, float b2 = 0.95f, float epsilon = 1e-8f, float wd = 0.01f)
        : params(parameters), grads(gradients), lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), weight_decay(wd), step_count(0) {
        for (auto p : params) {
            Tensor m_i(p->shape); m_i.zeros(); m.push_back(m_i);
            Tensor v_i(p->shape); v_i.zeros(); v.push_back(v_i);
        }
    }

    void step() override {
        step_count++;
        float bias_correction1 = 1.0f - std::pow(beta1, step_count);
        float bias_correction2 = 1.0f - std::pow(beta2, step_count);

        for (size_t i = 0; i < params.size(); ++i) {
            size_t sz = params[i]->total_size();
            for (size_t k = 0; k < sz; ++k) {
                float g = grads[i]->data[k];

                // 1. Momento de 1er orden (media) y 2do orden (varianza no centrada)
                m[i].data[k] = beta1 * m[i].data[k] + (1.0f - beta1) * g;
                v[i].data[k] = beta2 * v[i].data[k] + (1.0f - beta2) * g * g;

                // 2. Corrección de sesgo
                float m_hat = m[i].data[k] / bias_correction1;
                float v_hat = v[i].data[k] / bias_correction2;

                // 3. Weight Decay desacoplado (AdamW)
                if (params[i]->shape.size() >= 2 && weight_decay > 0.0f) {
                    params[i]->data[k] -= lr * weight_decay * params[i]->data[k];
                }

                // 4. Actualización del parámetro
                params[i]->data[k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }

    void zero_grad() override {
        for (auto g : grads) g->zeros();
    }
};

} // namespace ns

#endif // NEURAL_SUITE_OPTIMIZERS_H
