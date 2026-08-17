/**
 * @file layer.h
 * @brief Clase Base Abstracta Layer para la arquitectura modular de NeuralSuite.
 * @details Define el contrato polimórfico C++ que deben cumplir todas las capas (Linear, Conv2D, LSTM, Attention).
 */

#ifndef NEURAL_SUITE_LAYER_H
#define NEURAL_SUITE_LAYER_H

#include "tensor.h"

namespace ns {

/**
 * @class Layer
 * @brief Interfaz abstracta orientada a objetos para todas las capas de redes neuronales.
 */
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Paso hacia adelante (Forward Pass)
     * @param input Tensor de entrada a la capa
     * @return Tensor resultado de la transformación de la capa
     */
    virtual Tensor forward(const Tensor& input) = 0;

    /**
     * @brief Paso hacia atrás (Backward Pass) para propagación de gradientes
     * @param grad_output Derivada parcial dL/dY proveniente de la capa siguiente
     * @return Derivada parcial dL/dX para propagar a la capa anterior
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;

    /**
     * @brief Retorna punteros a los tensores de parámetros entrenables (pesos W, sesgos b)
     */
    virtual std::vector<Tensor*> get_parameters() { return {}; }

    /**
     * @brief Retorna punteros a los tensores de gradientes asociados (dW, db)
     */
    virtual std::vector<Tensor*> get_gradients() { return {}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LAYER_H
