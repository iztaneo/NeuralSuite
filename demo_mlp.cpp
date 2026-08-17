/**
 * @file demo_mlp.cpp
 * @brief Demostración de Entrenamiento de un Perceptrón Multicapa (MLP) en C++ puro.
 */

#include "tensor.h"
#include "layers/linear.h"
#include "activations.h"
#include "losses.h"
#include "optimizers.h"
#include <iostream>

int main() {
    std::cout << "============================================================\n" << std::flush;
    std::cout << "🧠 Demostración 1: Entrenamiento de Clasificador MLP en C++\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    // Problema XOR: 4 muestras, 2 características de entrada -> 2 clases de salida
    ns::Tensor X({4, 2});
    X.data[0] = 0.0f; X.data[1] = 0.0f;
    X.data[2] = 0.0f; X.data[3] = 1.0f;
    X.data[4] = 1.0f; X.data[5] = 0.0f;
    X.data[6] = 1.0f; X.data[7] = 1.0f;

    ns::Tensor Y({4});
    Y.data[0] = 0.0f; // 0 XOR 0 = 0
    Y.data[1] = 1.0f; // 0 XOR 1 = 1
    Y.data[2] = 1.0f; // 1 XOR 0 = 1
    Y.data[3] = 0.0f; // 1 XOR 1 = 0

    // Red MLP: Linear(2 -> 8) -> ReLU -> Linear(8 -> 2)
    ns::Linear fc1(2, 8);
    ns::Activation relu(ns::ActivationType::RELU);
    ns::Linear fc2(8, 2);

    ns::CrossEntropyLoss criterion;

    // Recolectar parámetros
    std::vector<ns::Tensor*> params, grads;
    for (auto p : fc1.get_parameters()) params.push_back(p);
    for (auto p : fc2.get_parameters()) params.push_back(p);
    for (auto g : fc1.get_gradients()) grads.push_back(g);
    for (auto g : fc2.get_gradients()) grads.push_back(g);

    ns::AdamW optimizer(params, grads, 0.05f);

    std::cout << "🏋️ Entrenando MLP durante 200 iteraciones...\n" << std::flush;
    for (int epoch = 1; epoch <= 200; ++epoch) {
        optimizer.zero_grad();

        // Forward
        ns::Tensor h1 = fc1.forward(X);
        ns::Tensor a1 = relu.forward(h1);
        ns::Tensor logits = fc2.forward(a1);

        float loss = criterion.forward(logits, Y);

        // Backward
        ns::Tensor dlogits = criterion.backward();
        ns::Tensor da1 = fc2.backward(dlogits);
        ns::Tensor dh1 = relu.backward(da1);
        fc1.backward(dh1);

        optimizer.step();

        if (epoch % 50 == 0 || epoch == 1) {
            std::cout << "Época " << epoch << " | Loss MLP: " << loss << "\n" << std::flush;
        }
    }

    std::cout << "✅ ¡Entrenamiento MLP completado exitosamente en C++!\n" << std::flush;
    return 0;
}
