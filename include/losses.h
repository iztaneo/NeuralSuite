/**
 * @file losses.h
 * @brief Funciones de Pérdida (CrossEntropyLoss, MSELoss) para entrenamiento de redes en C++ puro.
 */

#ifndef NEURAL_SUITE_LOSSES_H
#define NEURAL_SUITE_LOSSES_H

#include "tensor.h"

namespace ns {

/**
 * @class Loss
 * @brief Clase base para calcular el costo (loss) y los gradientes iniciales dL/dLogits.
 */
class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward() = 0;
};

/**
 * @class CrossEntropyLoss
 * @brief Entropía Cruzada con Softmax numéricamente estable.
 * @details Computa Loss = -log( softmax(logits)[target] ) y dLogits = prob - y_onehot.
 */
class CrossEntropyLoss : public Loss {
public:
    Tensor last_preds;
    Tensor last_targets;

    float forward(const Tensor& predictions, const Tensor& targets) override {
        last_preds = predictions;
        last_targets = targets;

        int N = predictions.shape[0];
        int C = predictions.shape[1];

        float total_loss = 0.0f;
        for (int i = 0; i < N; ++i) {
            int target_cls = static_cast<int>(targets.data[i]);
            
            float max_val = predictions.data[i * C];
            for (int c = 1; c < C; ++c) {
                if (predictions.data[i * C + c] > max_val) max_val = predictions.data[i * C + c];
            }

            float sum = 0.0f;
            for (int c = 0; c < C; ++c) sum += std::exp(predictions.data[i * C + c] - max_val);
            float prob = std::exp(predictions.data[i * C + target_cls] - max_val) / sum;

            total_loss -= std::log(prob + 1e-7f);
        }
        return total_loss / N;
    }

    Tensor backward() override {
        int N = last_preds.shape[0];
        int C = last_preds.shape[1];

        Tensor dlogits(last_preds.shape);

        for (int i = 0; i < N; ++i) {
            int target_cls = static_cast<int>(last_targets.data[i]);
            
            float max_val = last_preds.data[i * C];
            for (int c = 1; c < C; ++c) {
                if (last_preds.data[i * C + c] > max_val) max_val = last_preds.data[i * C + c];
            }

            float sum = 0.0f;
            for (int c = 0; c < C; ++c) sum += std::exp(last_preds.data[i * C + c] - max_val);

            for (int c = 0; c < C; ++c) {
                float prob = std::exp(last_preds.data[i * C + c] - max_val) / sum;
                float target_val = (c == target_cls) ? 1.0f : 0.0f;
                dlogits.data[i * C + c] = (prob - target_val) / N;
            }
        }
        return dlogits;
    }
};

/**
 * @class MSELoss
 * @brief Error Cuadrático Medio: L = (1/N) * sum( (y_pred - y_true)^2 )
 */
class MSELoss : public Loss {
public:
    Tensor last_preds;
    Tensor last_targets;

    float forward(const Tensor& predictions, const Tensor& targets) override {
        last_preds = predictions;
        last_targets = targets;
        size_t sz = predictions.total_size();

        float loss = 0.0f;
        for (size_t i = 0; i < sz; ++i) {
            float diff = predictions.data[i] - targets.data[i];
            loss += diff * diff;
        }
        return loss / sz;
    }

    Tensor backward() override {
        size_t sz = last_preds.total_size();
        Tensor dpreds(last_preds.shape);
        for (size_t i = 0; i < sz; ++i) {
            dpreds.data[i] = 2.0f * (last_preds.data[i] - last_targets.data[i]) / sz;
        }
        return dpreds;
    }
};

} // namespace ns

#endif // NEURAL_SUITE_LOSSES_H
