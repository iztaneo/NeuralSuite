// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de models/ocr.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "models/ocr.h"

namespace neuralsuite {

std::vector<std::string> CRNNModel::PartirUtf8(const std::string& texto) {
    std::vector<std::string> simbolos;
    for (size_t i = 0; i < texto.size();) {
      const unsigned char c = static_cast<unsigned char>(texto[i]);
      size_t largo = 1;
      if ((c & 0xE0) == 0xC0) largo = 2;
      else if ((c & 0xF0) == 0xE0) largo = 3;
      else if ((c & 0xF8) == 0xF0) largo = 4;
      largo = std::min(largo, texto.size() - i);
      simbolos.push_back(texto.substr(i, largo));
      i += largo;
    }
    return simbolos;
  }

std::vector<std::string> CRNNModel::DefaultVocab() {
    std::string texto;
    for (char c = 'A'; c <= 'Z'; ++c) texto += c;
    for (char c = 'a'; c <= 'z'; ++c) texto += c;
    for (char c = '0'; c <= '9'; ++c) texto += c;
    texto += " ";
    texto += "\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1";
    texto += "\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1";
    texto += ".,;:\u00bf?\u00a1!-\u2014()\'\"";
    return PartirUtf8(texto);
  }

Tensor CRNNModel::Forward(const Tensor& images) {
    const std::vector<int>& shape = images.Shape();
    if (shape.size() != 4) {
      throw std::invalid_argument("CRNNModel: la entrada debe ser [batch, canales, alto, ancho].");
    }
    if (shape[1] != in_channels_) {
      throw std::invalid_argument(
          "CRNNModel: la imagen tiene " + std::to_string(shape[1]) + " canales y el modelo espera " +
          std::to_string(in_channels_) + ".");
    }
    if (shape[2] != kInputHeight) {
      throw std::invalid_argument(
          "CRNNModel: el alto de la imagen es " + std::to_string(shape[2]) + " y debe ser " +
          std::to_string(kInputHeight) + ".");
    }
    if (shape[3] % kWidthReduction != 0) {
      throw std::invalid_argument(
          "CRNNModel: el ancho " + std::to_string(shape[3]) + " debe ser multiplo de " +
          std::to_string(kWidthReduction) + ".");
    }

    batch_size_ = shape[0];
    timesteps_ = shape[3] / kWidthReduction;

    Tensor h = pool1_.Forward(relu1_.Forward(conv1_.Forward(images)));
    h = pool2_.Forward(relu2_.Forward(conv2_.Forward(h)));
    h = pool3_.Forward(relu3_.Forward(conv3_.Forward(h)));

    // [B, 64, 1, T] -> [T, B, 64]: el BiLSTM recorre el tiempo en el eje 0.
    Tensor seq = MapToSequence(h, batch_size_, kSeqFeatures, timesteps_);
    seq = bilstm_.Forward(seq);

    // [T, B, 2H] -> [B, T, 2H]: la salida se indexa por muestra, como los
    // objetivos de la perdida.
    Tensor seq_bt = SwapFirstTwoAxes(seq, timesteps_, batch_size_, 2 * hidden_dim_);
    return fc_.Forward(seq_bt);
  }

Tensor CRNNModel::Backward(const Tensor& dout) {
    Tensor dseq_bt = fc_.Backward(dout);
    Tensor dseq = SwapFirstTwoAxes(dseq_bt, batch_size_, timesteps_, 2 * hidden_dim_);
    Tensor dfeat = bilstm_.Backward(dseq);
    Tensor dh = SequenceToMap(dfeat, batch_size_, kSeqFeatures, timesteps_);

    dh = conv3_.Backward(relu3_.Backward(pool3_.Backward(dh)));
    dh = conv2_.Backward(relu2_.Backward(pool2_.Backward(dh)));
    return conv1_.Backward(relu1_.Backward(pool1_.Backward(dh)));
  }

std::vector<std::string> CRNNModel::DecodeBatch(const Tensor& logits, const std::vector<std::string>& vocab, int clase_blanco) const {
    const std::vector<int>& shape = logits.Shape();
    if (shape.size() != 3) {
      throw std::invalid_argument("DecodeBatch espera logits de rango 3 [batch, tiempo, clases]");
    }
    const int batch = shape[0], steps = shape[1], classes = shape[2];

    std::vector<std::string> words;
    words.reserve(batch);
    for (int b = 0; b < batch; ++b) {
      std::string word;
      int prev = -1;
      for (int t = 0; t < steps; ++t) {
        const size_t base = static_cast<size_t>(b * steps + t) * classes;
        int best = 0;
        float best_val = logits[base];
        for (int c = 1; c < classes; ++c) {
          if (logits[base + c] > best_val) {
            best_val = logits[base + c];
            best = c;
          }
        }
        if (best != prev) {
          const bool relleno = (clase_blanco >= 0) ? (best == clase_blanco)
                                                   : (best < static_cast<int>(vocab.size()) &&
                                                      vocab[best] == " ");
          if (!relleno && best < static_cast<int>(vocab.size())) word += vocab[best];
          prev = best;
        }
      }
      words.push_back(std::move(word));
    }
    return words;
  }

std::string CRNNModel::DecodeWord(const Tensor& logits, const std::vector<std::string>& vocab, int clase_blanco) const {
    const std::vector<std::string> words = DecodeBatch(logits, vocab, clase_blanco);
    return words.empty() ? std::string() : words.front();
  }

bool CRNNModel::Save(const std::string& path) {
    const auto result = nsf::Save(path, nsf::FromNamedParameters(NamedParameters()),
                                  ArchitectureMetadata());
    if (!result) std::cerr << "Error al guardar: " << result.error << "\n";
    return result.ok;
  }

bool CRNNModel::Load(const std::string& path) {
    const auto result = nsf::Load(path, nsf::FromNamedParameters(NamedParameters()),
                                  ArchitectureMetadata());
    if (!result) std::cerr << "Error al cargar: " << result.error << "\n";
    return result.ok;
  }

std::map<std::string, std::string> CRNNModel::ArchitectureMetadata() const {
    return {{"arch", "crnn"},
            {"in_channels", std::to_string(in_channels_)},
            {"hidden_dim", std::to_string(hidden_dim_)},
            {"num_classes", std::to_string(num_classes_)}};
  }

Tensor CRNNModel::MapToSequence(const Tensor& h, int batch, int channels, int steps) {
    Tensor seq({steps, batch, channels});
    for (int t = 0; t < steps; ++t) {
      for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
          seq[static_cast<size_t>(t * batch + b) * channels + c] =
              h[static_cast<size_t>(b * channels + c) * steps + t];
        }
      }
    }
    return seq;
  }

Tensor CRNNModel::SequenceToMap(const Tensor& seq, int batch, int channels, int steps) {
    Tensor h({batch, channels, 1, steps});
    for (int t = 0; t < steps; ++t) {
      for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
          h[static_cast<size_t>(b * channels + c) * steps + t] =
              seq[static_cast<size_t>(t * batch + b) * channels + c];
        }
      }
    }
    return h;
  }

Tensor CRNNModel::SwapFirstTwoAxes(const Tensor& x, int d0, int d1, int k) {
    Tensor out({d1, d0, k});
    for (int i = 0; i < d0; ++i) {
      for (int j = 0; j < d1; ++j) {
        std::memcpy(out.Data() + static_cast<size_t>(j * d0 + i) * k,
                    x.Data() + static_cast<size_t>(i * d1 + j) * k, k * sizeof(float));
      }
    }
    return out;
  }

void SynthTextGenerator::Generate(Tensor& images, Tensor& targets, int batch, int word_len, int vocab_size, uint32_t seed, float noise) {
    const int width = word_len * kCharWidth;
    images.Resize({batch, 1, CRNNModel::kInputHeight, width});
    targets.Resize({batch, word_len});
    images.Zeros();

    uint32_t state = seed ? seed : 1u;
    auto next = [&state]() {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      return state;
    };

    for (int b = 0; b < batch; ++b) {
      for (int t = 0; t < word_len; ++t) {
        const int cls = static_cast<int>(next() % static_cast<uint32_t>(vocab_size));
        targets[static_cast<size_t>(b) * word_len + t] = static_cast<float>(cls);

        const uint32_t glyph = Glyph(cls);
        for (int cell = 0; cell < 32; ++cell) {
          if (((glyph >> cell) & 1u) == 0u) continue;
          const int band = cell / kCharWidth;     // cual de las 8 franjas verticales
          const int col = cell % kCharWidth;      // cual de las 4 columnas
          for (int r = 0; r < CRNNModel::kInputHeight / 8; ++r) {
            const int y = band * (CRNNModel::kInputHeight / 8) + r;
            const int x = t * kCharWidth + col;
            images[static_cast<size_t>(y) * width + x +
                   static_cast<size_t>(b) * CRNNModel::kInputHeight * width] = 1.0f;
          }
        }
      }
    }

    if (noise > 0.0f) {
      for (size_t i = 0; i < images.TotalSize(); ++i) {
        // Ruido uniforme centrado, para que la red no dependa de valores exactos.
        const float u = static_cast<float>(next() % 2001u) / 1000.0f - 1.0f;
        images[i] += noise * u;
      }
    }
  }

uint32_t SynthTextGenerator::Glyph(int cls) {
    // Mezcla entera de Knuth: reparte clases consecutivas en patrones que no se
    // parecen, que es lo unico que se le pide.
    uint32_t h = static_cast<uint32_t>(cls + 1) * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    // Al menos un bit encendido: una clase en blanco seria indistinguible del fondo.
    return h | (1u << (static_cast<uint32_t>(cls) % 32u));
  }

}  // namespace neuralsuite
