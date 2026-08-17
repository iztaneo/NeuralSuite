/**
 * @file generate_llm.cpp
 * @brief Inferencia y Generación Autorregresiva de Texto en C++17 puro.
 */

#include "tensor.h"
#include "tokenizer.h"
#include "gpt.h"
#include <iostream>

int main() {
    std::cout << "============================================================\n" << std::flush;
    std::cout << "🤖 Generación Autorregresiva de Texto en C++ (NeuralSuite)\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    ns::CharTokenizer tokenizer;
    if (!tokenizer.load("vocab_cpp.txt")) {
        std::cout << "⚠️ No se encontró 'vocab_cpp.txt'. Usando vocabulario por defecto...\n" << std::flush;
        tokenizer.build_vocab("First Citizen: Speak, speak. MENENIUS: What work's, my countrymen?");
    }

    ns::GPTConfig config;
    config.vocab_size = tokenizer.vocab_size;
    config.block_size = 32;
    config.n_layer = 2;
    config.n_head = 2;
    config.n_embd = 32;

    ns::GPTModel model(config);

    std::string prompt = "First Citizen:\n";
    std::cout << "Prompt de entrada: '" << prompt << "'\n\n" << std::flush;

    std::vector<int> tokens = tokenizer.encode(prompt);
    int max_new_tokens = 50;

    for (int step = 0; step < max_new_tokens; ++step) {
        int seq_len = static_cast<int>(tokens.size());
        int start_idx = (seq_len > config.block_size) ? (seq_len - config.block_size) : 0;
        int curr_len = seq_len - start_idx;

        ns::Tensor idx({1, curr_len});
        for (int i = 0; i < curr_len; ++i) {
            idx.data[i] = static_cast<float>(tokens[start_idx + i]);
        }

        ns::Tensor logits = model.forward(idx);

        // Extraer logit del último token
        int last_offset = (curr_len - 1) * config.vocab_size;
        int best_token = 0;
        float max_logit = logits.data[last_offset];

        for (int v = 1; v < config.vocab_size; ++v) {
            if (logits.data[last_offset + v] > max_logit) {
                max_logit = logits.data[last_offset + v];
                best_token = v;
            }
        }

        tokens.push_back(best_token);
    }

    std::string generated = tokenizer.decode(tokens);
    std::cout << "------------------------------------------------------------\n" << std::flush;
    std::cout << "Texto Generado por el LLM C++:\n" << generated << "\n" << std::flush;
    std::cout << "------------------------------------------------------------\n" << std::flush;

    return 0;
}
