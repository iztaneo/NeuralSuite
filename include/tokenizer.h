// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file tokenizer.h
 * @brief Tokenizadores de caracteres y de bytes.
 */

#ifndef NEURAL_SUITE_INCLUDE_TOKENIZER_H_
#define NEURAL_SUITE_INCLUDE_TOKENIZER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuralsuite {

/**
 * @class CharTokenizer
 * @brief Tokenizador por byte, con simbolo explicito para lo desconocido.
 *
 * El token 0 esta reservado para `<UNK>`. Antes no existia: un caracter fuera
 * del vocabulario se codificaba como token 0, que era un caracter valido, de
 * modo que el texto se corrompia sin aviso. Con el vocabulario {a, b, c},
 * codificar y decodificar "axc" devolvia "aac": la `x` desconocida se convertia
 * en `a`.
 *
 * @warning Trabaja sobre `char`, que en C++ es un byte y no un caracter. Un
 * texto UTF-8 queda fragmentado: `ñ` ocupa dos posiciones y el modelo nunca ve
 * el caracter completo. Para texto que no sea ASCII conviene `ByteTokenizer`,
 * que hace explicita esa realidad en lugar de disimularla.
 */
class CharTokenizer {
 public:
  /** @brief Identificador reservado para los simbolos fuera del vocabulario. */
  static constexpr int kUnknownToken = 0;

  CharTokenizer() = default;
  explicit CharTokenizer(const std::string& text);

  void BuildVocab(const std::string& text);
  [[nodiscard]] std::vector<int> Encode(const std::string& text) const;
  [[nodiscard]] std::string Decode(const std::vector<int>& tokens) const;

  bool Save(const std::string& filepath) const;
  bool Load(const std::string& filepath);

  /** @brief Tamano del vocabulario, incluido `<UNK>`. */
  [[nodiscard]] int VocabSize() const { return vocab_size_; }

  /** @brief Cuantos simbolos de `text` quedarian fuera del vocabulario. */
  [[nodiscard]] size_t CountUnknown(const std::string& text) const;

 private:
  std::vector<char> chars_;
  std::unordered_map<char, int> stoi_;
  std::unordered_map<int, char> itos_;
  int vocab_size_ = 0;
};

/**
 * @class ByteTokenizer
 * @brief Vocabulario fijo de 256 simbolos, uno por valor de byte.
 *
 * Cualquier texto es una secuencia de bytes, asi que este tokenizador **no
 * puede encontrarse con un simbolo desconocido**: la nocion misma de `<UNK>`
 * desaparece. Y como no construye vocabulario a partir de un corpus, el mismo
 * modelo sirve para cualquier idioma sin reentrenar el tokenizador.
 *
 * El precio es que un caracter no ASCII ocupa varios tokens —`ñ` son dos,
 * un emoji hasta cuatro—, de modo que las secuencias son mas largas. La
 * diferencia con `CharTokenizer` no es que aqui el texto se fragmente y alli
 * no: alli tambien se fragmenta, solo que lo llama "caracteres". Aqui al menos
 * la unidad esta declarada.
 *
 * Mantiene la dependencia cero: no hace falta ninguna libreria Unicode para
 * tratar bytes.
 */
class ByteTokenizer {
 public:
  static constexpr int kVocabSize = 256;

  [[nodiscard]] std::vector<int> Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) tokens.push_back(static_cast<int>(c));
    return tokens;
  }

  [[nodiscard]] std::string Decode(const std::vector<int>& tokens) const {
    std::string out;
    out.reserve(tokens.size());
    for (int t : tokens) {
      if (t < 0 || t >= kVocabSize) continue;  // fuera de rango: no es un byte
      out.push_back(static_cast<char>(static_cast<unsigned char>(t)));
    }
    return out;
  }

  [[nodiscard]] int VocabSize() const { return kVocabSize; }
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_TOKENIZER_H_
