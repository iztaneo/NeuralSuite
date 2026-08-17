/**
 * @file activations.h
 * @brief Capas de Activación (ReLU, GELU, Sigmoid, Tanh) para NeuralSuite.
 */

#ifndef NEURAL_SUITE_ACTIVATIONS_H
#define NEURAL_SUITE_ACTIVATIONS_H

#include "layer.h"

namespace ns {

enum class ActivationType { RELU, GELU, SIGMOID, TANH };

class Activation : public Layer {
public:
    ActivationType type;
    Tensor last_input;
    Tensor last_output;

    Activation(ActivationType act_type) : type(act_type) {}

    Tensor forward(const Tensor& input) override {
        last_input = input;
        Tensor output;
        switch (type) {
            case ActivationType::RELU:    relu_forward(input, output); break;
            case ActivationType::GELU:    gelu_forward(input, output); break;
            case ActivationType::SIGMOID: sigmoid_forward(input, output); break;
            case ActivationType::TANH:    tanh_forward(input, output); break;
        }
        last_output = output;
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        Tensor dx;
        switch (type) {
            case ActivationType::RELU:    relu_backward(dout, last_input, dx); break;
            case ActivationType::GELU:    gelu_backward(dout, last_input, dx); break;
            case ActivationType::SIGMOID: sigmoid_backward(dout, last_output, dx); break;
            case ActivationType::TANH:    tanh_backward(dout, last_output, dx); break;
        }
        return dx;
    }
};

} // namespace ns

#endif // NEURAL_SUITE_ACTIVATIONS_H
