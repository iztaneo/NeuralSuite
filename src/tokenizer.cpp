#include "tokenizer.h"

namespace ns {

CharTokenizer::CharTokenizer(const std::string& text) {
    build_vocab(text);
}

void CharTokenizer::build_vocab(const std::string& text) {
    std::unordered_map<char, bool> seen;
    chars.clear();
    stoi.clear();
    itos.clear();

    for (char c : text) {
        if (!seen[c]) {
            seen[c] = true;
            chars.push_back(c);
        }
    }

    std::sort(chars.begin(), chars.end());
    vocab_size = static_cast<int>(chars.size());

    for (int i = 0; i < vocab_size; ++i) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        auto it = stoi.find(c);
        if (it != stoi.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(0); // Token de fallback
        }
    }
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (int t : tokens) {
        auto it = itos.find(t);
        if (it != itos.end()) {
            text.push_back(it->second);
        } else {
            text.push_back('?');
        }
    }
    return text;
}

bool CharTokenizer::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;
    out << vocab_size << "\n";
    for (int i = 0; i < vocab_size; ++i) {
        out << (int)(unsigned char)chars[i] << "\n";
    }
    return true;
}

bool CharTokenizer::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) return false;
    in >> vocab_size;
    chars.clear();
    stoi.clear();
    itos.clear();
    for (int i = 0; i < vocab_size; ++i) {
        int code;
        in >> code;
        char c = (char)code;
        chars.push_back(c);
        stoi[c] = i;
        itos[i] = c;
    }
    return true;
}

} // namespace ns
