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

  // El indice 0 queda reservado para `<UNK>`, asi que los simbolos reales
  // empiezan en 1. Antes ocupaban desde el 0 y un caracter desconocido se
  // confundia con el primero del vocabulario.
  vocab_size_ = static_cast<int>(chars_.size()) + 1;
  for (size_t i = 0; i < chars_.size(); ++i) {
    const int id = static_cast<int>(i) + 1;
    stoi_[chars_[i]] = id;
    itos_[id] = chars_[i];
  }
}

std::vector<int> CharTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.reserve(text.size());
  for (char c : text) {
    auto it = stoi_.find(c);
    tokens.push_back(it != stoi_.end() ? it->second : kUnknownToken);
  }
  return tokens;
}

std::string CharTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  text.reserve(tokens.size());
  for (int t : tokens) {
    auto it = itos_.find(t);
    // Lo desconocido se marca de forma visible en vez de sustituirse por un
    // caracter cualquiera del vocabulario.
    text.push_back(it != itos_.end() ? it->second : '?');
  }
  return text;
}

size_t CharTokenizer::CountUnknown(const std::string& text) const {
  size_t n = 0;
  for (char c : text) {
    if (stoi_.find(c) == stoi_.end()) ++n;
  }
  return n;
}

bool CharTokenizer::Save(const std::string& filepath) const {
  std::ofstream out(filepath);
  if (!out.is_open()) return false;
  // Se guardan solo los simbolos reales; `<UNK>` es implicito y siempre ocupa
  // el indice 0, de modo que el archivo no depende de esa convencion.
  out << chars_.size() << "\n";
  for (char c : chars_) {
    out << static_cast<int>(static_cast<unsigned char>(c)) << "\n";
  }
  return true;
}

bool CharTokenizer::Load(const std::string& filepath) {
  std::ifstream in(filepath);
  if (!in.is_open()) return false;

  int declared_size = 0;
  if (!(in >> declared_size)) return false;

  // El tamano lo dicta el archivo y se usaba directamente como limite de bucle:
  // un valor corrupto o adversarial dejaba el proceso colgado. Un vocabulario
  // de bytes no puede exceder 256 simbolos distintos.
  constexpr int kMaxVocabSize = 256;
  if (declared_size < 0 || declared_size > kMaxVocabSize) return false;

  chars_.clear();
  stoi_.clear();
  itos_.clear();

  for (int i = 0; i < declared_size; ++i) {
    int code;
    // Un archivo truncado dejaba el resto del vocabulario sin inicializar.
    if (!(in >> code)) {
      chars_.clear();
      stoi_.clear();
      itos_.clear();
      vocab_size_ = 0;
      return false;
    }
    const char c = static_cast<char>(code);
    chars_.push_back(c);
    const int id = i + 1;   // el 0 sigue siendo `<UNK>`
    stoi_[c] = id;
    itos_[id] = c;
  }

  vocab_size_ = declared_size + 1;
  return true;
}

}  // namespace neuralsuite
