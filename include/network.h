/**
 * @file network.h
 * @brief Contenedor Secuencial de Capas para redes neuronales en NeuralSuite.
 */

#ifndef NEURAL_SUITE_NETWORK_H
#define NEURAL_SUITE_NETWORK_H

#include "layer.h"
#include <memory>
#include <vector>

namespace ns {

class Sequential {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    void add(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    Tensor forward(const Tensor& input) {
        Tensor current = input;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    Tensor backward(const Tensor& grad_output) {
        Tensor current_grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
        return current_grad;
    }

    std::vector<Tensor*> get_parameters() {
        std::vector<Tensor*> all_params;
        for (auto& layer : layers) {
            auto p = layer->get_parameters();
            all_params.insert(all_params.end(), p.begin(), p.end());
        }
        return all_params;
    }

    std::vector<Tensor*> get_gradients() {
        std::vector<Tensor*> all_grads;
        for (auto& layer : layers) {
            auto g = layer->get_gradients();
            all_grads.insert(all_grads.end(), g.begin(), g.end());
        }
        return all_grads;
    }
};

} // namespace ns

#endif // NEURAL_SUITE_NETWORK_H
