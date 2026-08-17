// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

#include "tokenizer.h"

namespace neuralsuite {

CharTokenizer::CharTokenizer(const std::string& text) {
  BuildVocab(text);
}

void CharTokenizer::BuildVocab(const std::string& text) {
  std::unordered_map<char, bool> seen;
  chars_.clear();
  stoi_.clear();
  itos_.clear();

  for (char c : text) {
    if (!seen[c]) {
      seen[c] = true;
      chars_.push_back(c);
    }
  }

  std::sort(chars_.begin(), chars_.end());
  vocab_size_ = static_cast<int>(chars_.size());

  for (int i = 0; i < vocab_size_; ++i) {
    stoi_[chars_[i]] = i;
    itos_[i] = chars_[i];
  }
}

std::vector<int> CharTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.reserve(text.size());
  for (char c : text) {
    auto it = stoi_.find(c);
    if (it != stoi_.end()) {
      tokens.push_back(it->second);
    } else {
      tokens.push_back(0);
    }
  }
  return tokens;
}

std::string CharTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  text.reserve(tokens.size());
  for (int t : tokens) {
    auto it = itos_.find(t);
    if (it != itos_.end()) {
      text.push_back(it->second);
    } else {
      text.push_back('?');
    }
  }
  return text;
}

bool CharTokenizer::Save(const std::string& filepath) const {
  std::ofstream out(filepath);
  if (!out.is_open()) return false;
  out << vocab_size_ << "\n";
  for (int i = 0; i < vocab_size_; ++i) {
    out << static_cast<int>(static_cast<unsigned char>(chars_[i])) << "\n";
  }
  return true;
}

bool CharTokenizer::Load(const std::string& filepath) {
  std::ifstream in(filepath);
  if (!in.is_open()) return false;
  in >> vocab_size_;
  chars_.clear();
  stoi_.clear();
  itos_.clear();
  for (int i = 0; i < vocab_size_; ++i) {
    int code;
    in >> code;
    char c = static_cast<char>(code);
    chars_.push_back(c);
    stoi_[c] = i;
    itos_[i] = c;
  }
  return true;
}

}  // namespace neuralsuite
