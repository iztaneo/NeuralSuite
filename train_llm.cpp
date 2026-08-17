/**
 * @file train_llm.cpp
 * @brief Entrenamiento del LLM Transformer (Decoder-Only) en C++17 puro.
 */

#include "tensor.h"
#include "tokenizer.h"
#include "gpt.h"
#include "optimizers.h"
#include "losses.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

int main(int argc, char** argv) {
    std::string data_path = "sample_data/input.txt";
    if (argc > 1) {
        data_path = argv[1];
    } else if (!std::ifstream(data_path).good() && std::ifstream("../sample_data/input.txt").good()) {
        data_path = "../sample_data/input.txt";
    }

    std::cout << "============================================================\n" << std::flush;
    std::cout << "🚀 Iniciando Entrenamiento del LLM en C++ (NeuralSuite)\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    // 1. Cargar dataset de texto
    std::ifstream file(data_path);

    if (!file.is_open()) {
        std::cerr << "❌ No se pudo abrir el archivo de datos: " << data_path << "\n" << std::flush;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    std::cout << "📄 Dataset cargado: " << text.size() << " caracteres.\n" << std::flush;

    // 2. Inicializar Tokenizador
    ns::CharTokenizer tokenizer(text);
    std::cout << "🔤 Vocabulario del tokenizador C++: " << tokenizer.vocab_size << " caracteres únicos.\n" << std::flush;
    tokenizer.save("vocab_cpp.txt");

    // 3. Tokenizar texto completo
    std::vector<int> tokens = tokenizer.encode(text);

    // 4. Configurar GPT C++
    ns::GPTConfig config;
    config.vocab_size = tokenizer.vocab_size;
    config.block_size = 32;
    config.n_layer = 2;
    config.n_head = 2;
    config.n_embd = 32;

    ns::GPTModel model(config);
    std::cout << "🧠 Modelo GPT C++ Creado exitosamente.\n" << std::flush;

    // 5. Configurar Optimizador
    std::vector<ns::Tensor*> params = model.get_parameters();
    std::vector<ns::Tensor*> grads = model.get_gradients();
    ns::AdamW optimizer(params, grads, 0.001f);
    ns::CrossEntropyLoss criterion;

    int max_iters = 50;
    int batch_size = 4;
    int block_size = config.block_size;

    std::cout << "🏋️ Entrenando durante " << max_iters << " iteraciones en C++...\n" << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 1; iter <= max_iters; ++iter) {
        optimizer.zero_grad();

        // Crear Batch (X, Y)
        ns::Tensor X({batch_size, block_size});
        ns::Tensor Y({batch_size * block_size});

        for (int b = 0; b < batch_size; ++b) {
            int start_idx = (iter * 17 + b * 13) % (tokens.size() - block_size - 1);
            for (int t = 0; t < block_size; ++t) {
                X.data[b * block_size + t] = static_cast<float>(tokens[start_idx + t]);
                Y.data[b * block_size + t] = static_cast<float>(tokens[start_idx + t + 1]);
            }
        }

        // Forward
        ns::Tensor logits = model.forward(X);
        ns::Tensor logits_2d({batch_size * block_size, config.vocab_size});
        std::memcpy(logits_2d.data, logits.data, logits.total_size() * sizeof(float));

        float loss = criterion.forward(logits_2d, Y);
        optimizer.step();

        if (iter % 10 == 0 || iter == max_iters) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            std::cout << "Step " << iter << "/" << max_iters << " | Loss LLM: " << loss << " | Tiempo: " << elapsed << "s\n" << std::flush;
        }
    }

    std::cout << "✅ ¡Entrenamiento del LLM en C++ completado exitosamente!\n" << std::flush;
    return 0;
}
