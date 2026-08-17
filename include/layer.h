/**
 * @file layer.h
 * @brief Clase Base Abstracta Layer para la arquitectura modular de NeuralSuite.
 */

#ifndef NEURAL_SUITE_LAYER_H
#define NEURAL_SUITE_LAYER_H

#include "tensor.h"

namespace ns {

/**
 * @class Layer
 * @brief Interfaz abstracta orientada a objetos para todas las capas de redes neuronales (MLP, CNN, LSTM, Attention).
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Paso hacia adelante (Forward Pass)
     */
    virtual Tensor forward(const Tensor& input) = 0;

    /**
     * @brief Paso hacia atrás (Backward Pass) para propagación de gradientes
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;

    /**
     * @brief Retorna punteros a los tensores de parámetros entrenables (pesos, sesgos)
     */
    virtual std::vector<Tensor*> get_parameters() { return {}; }

    /**
     * @brief Retorna punteros a los tensores de gradientes
     */
    virtual std::vector<Tensor*> get_gradients() { return {}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LAYER_H
