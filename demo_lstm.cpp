/**
 * @file demo_lstm.cpp
 * @brief Demostración de Entrenamiento de Red Recurrente (LSTM) en C++ puro.
 */

#include "tensor.h"
#include "layers/lstm.h"
#include "layers/linear.h"
#include "losses.h"
#include "optimizers.h"
#include <iostream>

int main() {
    std::cout << "============================================================\n" << std::flush;
    std::cout << "🔄 Demostración 3: Red Recurrente (LSTM) en C++\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    // Entrada: Secuencia de 5 pasos temporales, batch=1, input_dim=4
    ns::Tensor X({5, 1, 4});
    X.random_normal(0.0f, 1.0f);

    ns::Tensor Y({5, 1, 2});
    Y.zeros();

    ns::LSTM lstm(4, 8);
    ns::Linear fc(8, 2);

    ns::MSELoss criterion;

    std::vector<ns::Tensor*> params, grads;
    for (auto p : lstm.get_parameters()) params.push_back(p);
    for (auto p : fc.get_parameters()) params.push_back(p);
    for (auto g : lstm.get_gradients()) grads.push_back(g);
    for (auto g : fc.get_gradients()) grads.push_back(g);

    ns::AdamW optimizer(params, grads, 0.01f);

    std::cout << "🏋️ Entrenando LSTM durante 10 iteraciones...\n" << std::flush;
    for (int epoch = 1; epoch <= 10; ++epoch) {
        optimizer.zero_grad();

        ns::Tensor h_lstm = lstm.forward(X);
        
        ns::Tensor h_2d({5 * 1, 8});
        std::memcpy(h_2d.data, h_lstm.data, h_lstm.total_size() * sizeof(float));

        ns::Tensor logits_2d = fc.forward(h_2d);

        float loss = criterion.forward(logits_2d, Y);

        ns::Tensor dlogits = criterion.backward();
        ns::Tensor dh_2d = fc.backward(dlogits);

        ns::Tensor dh_lstm(h_lstm.shape);
        std::memcpy(dh_lstm.data, dh_2d.data, dh_2d.total_size() * sizeof(float));

        lstm.backward(dh_lstm);
        optimizer.step();

        std::cout << "Época " << epoch << " | Loss Recurrente LSTM: " << loss << "\n" << std::flush;
    }

    std::cout << "✅ ¡Entrenamiento LSTM completado exitosamente en C++!\n" << std::flush;
    return 0;
}
