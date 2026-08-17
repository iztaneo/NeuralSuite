// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file tokenizer.h
 * @brief Pure C++ Character Tokenizer following Google C++ Style Guide.
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
 * @brief Character-level Tokenizer for LLM Training and Inference.
 */
class CharTokenizer {
 public:
  CharTokenizer() = default;
  explicit CharTokenizer(const std::string& text);

  void BuildVocab(const std::string& text);
  [[nodiscard]] std::vector<int> Encode(const std::string& text) const;
  [[nodiscard]] std::string Decode(const std::vector<int>& tokens) const;

  bool Save(const std::string& filepath) const;
  bool Load(const std::string& filepath);

  [[nodiscard]] int VocabSize() const { return vocab_size_; }

 private:
  std::vector<char> chars_;
  std::unordered_map<char, int> stoi_;
  std::unordered_map<int, char> itos_;
  int vocab_size_ = 0;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_TOKENIZER_H_
