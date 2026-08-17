/**
 * @file tokenizer.h
 * @brief Tokenizador a nivel de caracteres en C++ puro para el LLM.
 */

#ifndef NEURAL_SUITE_TOKENIZER_H
#define NEURAL_SUITE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace ns {

class CharTokenizer {
public:
    std::vector<char> chars;
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    int vocab_size = 0;

    CharTokenizer() = default;
    CharTokenizer(const std::string& text);

    void build_vocab(const std::string& text);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);
};

} // namespace ns

#endif // NEURAL_SUITE_TOKENIZER_H
