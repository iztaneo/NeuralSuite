/**
 * @file linear.h
 * @brief Capa Densa (Fully Connected / Linear Layer): Y = X * W + b
 * @details Realiza una transformación afín lineal en el espacio vectorial.
 */

#ifndef NEURAL_SUITE_LINEAR_H
#define NEURAL_SUITE_LINEAR_H

#include "../layer.h"

namespace ns {

/**
 * @class Linear
 * @brief Capa Densa Fully Connected.
 * @details Multiplica la matriz de entrada X [B, in_features] por la matriz de pesos W [in_features, out_features]
 * y le suma el vector de sesgos b [out_features].
 */
class Linear : public Layer {
public:
    int in_features;   ///< Dimensión de las características de entrada
    int out_features;  ///< Dimensión de las características de salida

    Tensor weight;     ///< Matriz de pesos entrenables [in_features, out_features]
    Tensor bias;       ///< Vector de sesgos entrenables [out_features]
    Tensor dweight;    ///< Acumulador de gradientes de pesos [in_features, out_features]
    Tensor dbias;      ///< Acumulador de gradientes de sesgos [out_features]
    Tensor last_input; ///< Caché de la entrada para el cálculo del gradiente en el paso backward

    /**
     * @brief Construye una capa densa lineal con inicialización Xavier/Glorot.
     * @param in_dim Dimensión de entrada (fan_in)
     * @param out_dim Dimensión de salida (fan_out)
     */
    Linear(int in_dim, int out_dim)
        : in_features(in_dim), out_features(out_dim),
          weight({in_dim, out_dim}), bias({out_dim}),
          dweight({in_dim, out_dim}), dbias({out_dim}) {
        weight.xavier_init(in_dim, out_dim);
        bias.zeros();
    }

    /**
     * @brief Computa Y = X * W + b
     * @param input Tensor de entrada [B, in_features]
     * @return Tensor de salida [B, out_features]
     */
    Tensor forward(const Tensor& input) override {
        last_input = input;
        int B = input.shape[0];
        Tensor output({B, out_features});
        
        // Multiplicación matricial paralela GEMM: output = input * weight
        matmul(input, weight, output);

        // Suma Broadcasting del vector de sesgos
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < out_features; ++j) {
                output.data[i * out_features + j] += bias.data[j];
            }
        }
        return output;
    }

    /**
     * @brief Computa los gradientes analíticos por la regla de la cadena:
     * dx = dout * W^T
     * dW = X^T * dout
     * db = sum_batch(dout)
     * @param dout Gradiente de la salida dL/dY [B, out_features]
     * @return Gradiente de la entrada dL/dX [B, in_features]
     */
    Tensor backward(const Tensor& dout) override {
        int B = last_input.shape[0];
        Tensor dx({B, in_features});

        // 1. dx = dout * W^T
        Tensor W_T = transpose(weight);
        matmul(dout, W_T, dx);

        // 2. dW = X^T * dout
        Tensor X_T = transpose(last_input);
        matmul(X_T, dout, dweight);

        // 3. db = sum_batch(dout)
        dbias.zeros();
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < out_features; ++j) {
                dbias.data[j] += dout.data[i * out_features + j];
            }
        }
        return dx;
    }

    /** @brief Retorna referencias a los parámetros entrenables {weight, bias} */
    std::vector<Tensor*> get_parameters() override { return {&weight, &bias}; }

    /** @brief Retorna referencias a los gradientes {dweight, dbias} */
    std::vector<Tensor*> get_gradients() override { return {&dweight, &dbias}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LINEAR_H
