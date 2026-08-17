/**
 * @file demo_cnn.cpp
 * @brief Demostración de Entrenamiento de Red Convolucional (CNN 2D) en C++ puro.
 */

#include "tensor.h"
#include "layers/conv2d.h"
#include "layers/maxpool2d.h"
#include "layers/linear.h"
#include "activations.h"
#include "losses.h"
#include "optimizers.h"
#include <iostream>

int main() {
    std::cout << "============================================================\n" << std::flush;
    std::cout << "🖼️ Demostración 2: Red Convolucional (CNN 2D) en C++\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    // Entrada: 2 imágenes de 1 canal (1 x 8 x 8)
    ns::Tensor X({2, 1, 8, 8});
    X.random_normal(0.0f, 1.0f);

    ns::Tensor Y({2});
    Y.data[0] = 0.0f;
    Y.data[1] = 1.0f;

    // Arquitectura CNN: Conv2D(1 -> 4, k=3) -> ReLU -> MaxPool2D(2x2) -> Linear(4*3*3 -> 2)
    ns::Conv2D conv(1, 4, 3, 1, 0); // Out: 4 x 6 x 6
    ns::Activation relu(ns::ActivationType::RELU);
    ns::MaxPool2D pool(2, 2);      // Out: 4 x 3 x 3
    ns::Linear fc(4 * 3 * 3, 2);

    ns::CrossEntropyLoss criterion;

    std::vector<ns::Tensor*> params, grads;
    for (auto p : conv.get_parameters()) params.push_back(p);
    for (auto p : fc.get_parameters()) params.push_back(p);
    for (auto g : conv.get_gradients()) grads.push_back(g);
    for (auto g : fc.get_gradients()) grads.push_back(g);

    ns::AdamW optimizer(params, grads, 0.01f);

    std::cout << "🏋️ Entrenando CNN durante 20 iteraciones...\n" << std::flush;
    for (int epoch = 1; epoch <= 20; ++epoch) {
        optimizer.zero_grad();

        ns::Tensor h_conv = conv.forward(X);
        ns::Tensor h_relu = relu.forward(h_conv);
        ns::Tensor h_pool = pool.forward(h_relu);

        // Flatten: [2, 4, 3, 3] -> [2, 36]
        ns::Tensor h_flat({2, 4 * 3 * 3});
        std::memcpy(h_flat.data, h_pool.data, h_pool.total_size() * sizeof(float));

        ns::Tensor logits = fc.forward(h_flat);

        float loss = criterion.forward(logits, Y);

        ns::Tensor dlogits = criterion.backward();
        ns::Tensor dh_flat = fc.backward(dlogits);

        ns::Tensor dh_pool(h_pool.shape);
        std::memcpy(dh_pool.data, dh_flat.data, dh_flat.total_size() * sizeof(float));

        ns::Tensor dh_relu = pool.backward(dh_pool);
        ns::Tensor dh_conv = relu.backward(dh_relu);
        conv.backward(dh_conv);

        optimizer.step();

        if (epoch % 5 == 0 || epoch == 1) {
            std::cout << "Época " << epoch << " | Loss Convolucional: " << loss << "\n" << std::flush;
        }
    }

    std::cout << "✅ ¡Entrenamiento CNN completado exitosamente en C++!\n" << std::flush;
    return 0;
}
