/**
 * @file tokenizer.h
 * @brief Tokenizador de caracteres nativo en C++ puro para el LLM.
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

/**
 * @class CharTokenizer
 * @brief Transforma cadenas de texto en listas de enteros (tokens) y viceversa a nivel de caracteres.
 */
class CharTokenizer {
public:
    std::vector<char> chars;                     ///< Alfabeto ordenado de caracteres únicos
    std::unordered_map<char, int> stoi;          ///< Mapeo String to ID (carácter -> entero)
    std::unordered_map<int, char> itos;          ///< Mapeo ID to String (entero -> carácter)
    int vocab_size = 0;                          ///< Tamaño del vocabulario V

    CharTokenizer() = default;

    /** @brief Construye el vocabulario a partir de un texto fuente de entrenamiento */
    CharTokenizer(const std::string& text);

    /** @brief Extrae todos los caracteres únicos ordenados y crea las tablas stoi e itos */
    void build_vocab(const std::string& text);

    /** @brief Codifica un string en un vector de enteros (tokens) */
    std::vector<int> encode(const std::string& text) const;

    /** @brief Decodifica un vector de enteros en la cadena de texto original */
    std::string decode(const std::vector<int>& tokens) const;

    /** @brief Guarda el vocabulario codificado en un archivo de texto en disco */
    bool save(const std::string& filepath) const;

    /** @brief Carga un vocabulario persistido previamente desde disco */
    bool load(const std::string& filepath);
};

} // namespace ns

#endif // NEURAL_SUITE_TOKENIZER_H
