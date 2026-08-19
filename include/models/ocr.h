// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file ocr.h
 * @brief CRNN (Convolutional Recurrent Neural Network) para reconocimiento de texto.
 *
 * La red lee una imagen de una linea de texto y devuelve, para cada columna de
 * la imagen, la distribucion sobre el vocabulario. La arquitectura corresponde a
 * la implementacion de referencia en PyTorch (`LLMRasec/src/ocr.py`):
 *
 *     [B, 1, 32, W]
 *       Conv2D(1 -> 16, 3x3, pad 1) + ReLU + MaxPool 2x2   -> [B, 16, 16, W/2]
 *       Conv2D(16 -> 32, 3x3, pad 1) + ReLU + MaxPool 2x2  -> [B, 32,  8, W/4]
 *       Conv2D(32 -> 64, 3x3, pad 1) + ReLU + MaxPool 8x1  -> [B, 64,  1, W/4]
 *       BiLSTM(64 -> 2*hidden)                             -> [B, W/4, 2*hidden]
 *       Linear(2*hidden -> num_classes)                    -> [B, W/4, num_classes]
 *
 * El ultimo pooling colapsa el alto entero y deja el ancho intacto: cada
 * columna que sobrevive es un paso de la secuencia. Por eso la red devuelve
 * `W/4` predicciones y no una sola, y por eso puede leer una palabra completa
 * en vez de clasificar un unico caracter.
 *
 * El `BiLSTM` es lo que da sentido a la parte recurrente: una letra se
 * interpreta mejor sabiendo con que continua, no solo con que empieza. Una
 * version anterior de este archivo era `Conv2D -> MaxPool -> Linear` y producia
 * una prediccion por imagen, de modo que no podia leer una palabra ni usaba
 * contexto alguno.
 */

#ifndef NEURAL_SUITE_INCLUDE_MODELS_OCR_H_
#define NEURAL_SUITE_INCLUDE_MODELS_OCR_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "../activations.h"
#include "../layer.h"
#include "../layers/conv2d.h"
#include "../layers/linear.h"
#include "../layers/lstm.h"
#include "../layers/maxpool2d.h"
#include "../serialization.h"
#include "../tensor.h"

namespace neuralsuite {

/**
 * @class CRNNModel
 * @brief Reconocimiento de una linea de texto: imagen -> secuencia de caracteres.
 */
class CRNNModel : public Layer {
 public:
  /** @brief Alto que espera la red. El ultimo pooling lo colapsa de una vez. */
  static constexpr int kInputHeight = 32;
  /** @brief Cuanto se reduce el ancho: dos pooling de 2, el tercero lo respeta. */
  static constexpr int kWidthReduction = 4;

  /** @brief Vocabulario de la implementacion de referencia: 63 simbolos. */
  static std::vector<char> DefaultVocab() {
    std::vector<char> vocab;
    for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
    for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
    for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);
    vocab.push_back(' ');
    return vocab;
  }

  CRNNModel(int in_channels, int hidden_dim, int num_classes)
      : in_channels_(in_channels),
        hidden_dim_(hidden_dim),
        num_classes_(num_classes),
        conv1_(in_channels, 16, 3, 1, 1),
        relu1_(ActivationType::kRelu),
        pool1_(2, 2),
        conv2_(16, 32, 3, 1, 1),
        relu2_(ActivationType::kRelu),
        pool2_(2, 2),
        conv3_(32, 64, 3, 1, 1),
        relu3_(ActivationType::kRelu),
        // Ventana rectangular: colapsa las 8 filas que quedan y no toca el ancho.
        pool3_(kInputHeight / 4, 1, kInputHeight / 4, 1),
        bilstm_(64, hidden_dim),
        fc_(2 * hidden_dim, num_classes) {
    if (in_channels <= 0 || hidden_dim <= 0 || num_classes <= 0) {
      throw std::invalid_argument("CRNNModel: los tres tamanos deben ser positivos.");
    }
    // Una activacion por sitio, no una compartida. `Activation` guarda su
    // entrada para el backward, asi que reutilizar la misma instancia en dos
    // puntos de la red hace que la segunda pisada borre la cache de la primera
    // y el gradiente vuelva derivado en el punto equivocado.
    Register(&conv1_, "conv1");
    Register(&conv2_, "conv2");
    Register(&conv3_, "conv3");
    Register(&bilstm_, "bilstm");
    Register(&fc_, "fc");
  }

  /** @brief `[B, C, 32, W]` -> `[B, W/4, num_classes]`. */
  Tensor Forward(const Tensor& images) override {
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

  /** @brief `[B, T, num_classes]` -> `[B, C, 32, W]`. */
  Tensor Backward(const Tensor& dout) override {
    Tensor dseq_bt = fc_.Backward(dout);
    Tensor dseq = SwapFirstTwoAxes(dseq_bt, batch_size_, timesteps_, 2 * hidden_dim_);
    Tensor dfeat = bilstm_.Backward(dseq);
    Tensor dh = SequenceToMap(dfeat, batch_size_, kSeqFeatures, timesteps_);

    dh = conv3_.Backward(relu3_.Backward(pool3_.Backward(dh)));
    dh = conv2_.Backward(relu2_.Backward(pool2_.Backward(dh)));
    return conv1_.Backward(relu1_.Backward(pool1_.Backward(dh)));
  }

  /**
   * @brief Convierte los logits en una cadena por cada imagen del lote.
   *
   * Colapsa repeticiones consecutivas del mismo simbolo y descarta el espacio,
   * igual que `decode_word` en la implementacion de referencia. Es un decodificado
   * voraz: no busca la secuencia mas probable en conjunto, solo el mejor
   * simbolo en cada paso.
   */
  [[nodiscard]] std::vector<std::string> DecodeBatch(const Tensor& logits,
                                                     const std::vector<char>& vocab) const {
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
          if (best < static_cast<int>(vocab.size()) && vocab[best] != ' ') word += vocab[best];
          prev = best;
        }
      }
      words.push_back(std::move(word));
    }
    return words;
  }

  /** @brief La primera imagen del lote, para el caso de una sola linea. */
  [[nodiscard]] std::string DecodeWord(const Tensor& logits,
                                       const std::vector<char>& vocab) const {
    const std::vector<std::string> words = DecodeBatch(logits, vocab);
    return words.empty() ? std::string() : words.front();
  }

  bool Save(const std::string& path) {
    const auto result = nsf::Save(path, nsf::FromNamedParameters(NamedParameters()),
                                  ArchitectureMetadata());
    if (!result) std::cerr << "Error al guardar: " << result.error << "\n";
    return result.ok;
  }

  bool Load(const std::string& path) {
    const auto result = nsf::Load(path, nsf::FromNamedParameters(NamedParameters()),
                                  ArchitectureMetadata());
    if (!result) std::cerr << "Error al cargar: " << result.error << "\n";
    return result.ok;
  }

  /** @brief Descripcion que viaja dentro del archivo de pesos. */
  [[nodiscard]] std::map<std::string, std::string> ArchitectureMetadata() const {
    return {{"arch", "crnn"},
            {"in_channels", std::to_string(in_channels_)},
            {"hidden_dim", std::to_string(hidden_dim_)},
            {"num_classes", std::to_string(num_classes_)}};
  }

  [[nodiscard]] int NumClasses() const { return num_classes_; }
  /** @brief Cuantas predicciones devuelve la red para un ancho dado. */
  [[nodiscard]] static int TimestepsFor(int width) { return width / kWidthReduction; }

 private:
  /** @brief Canales que salen del extractor convolucional. */
  static constexpr int kSeqFeatures = 64;

  /** @brief `[B, C, 1, T]` -> `[T, B, C]`. */
  static Tensor MapToSequence(const Tensor& h, int batch, int channels, int steps) {
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

  /** @brief La inversa de MapToSequence, para el gradiente. */
  static Tensor SequenceToMap(const Tensor& seq, int batch, int channels, int steps) {
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

  /**
   * @brief Intercambia los dos primeros ejes de un `[d0, d1, k]`.
   *
   * Sirve en los dos sentidos —de `[T, B, K]` a `[B, T, K]` y al reves— porque
   * intercambiar dos ejes es su propia inversa: basta pasar los tamanos en el
   * orden que tiene el tensor de entrada. Estaban escritas como dos funciones
   * distintas, y no lo eran: calculaban exactamente lo mismo, de modo que
   * confundir una con otra no daba error pero invitaba a creer que si lo daria.
   */
  static Tensor SwapFirstTwoAxes(const Tensor& x, int d0, int d1, int k) {
    Tensor out({d1, d0, k});
    for (int i = 0; i < d0; ++i) {
      for (int j = 0; j < d1; ++j) {
        std::memcpy(out.Data() + static_cast<size_t>(j * d0 + i) * k,
                    x.Data() + static_cast<size_t>(i * d1 + j) * k, k * sizeof(float));
      }
    }
    return out;
  }

  int in_channels_;
  int hidden_dim_;
  int num_classes_;
  int batch_size_ = 0;
  int timesteps_ = 0;

  Conv2D conv1_;
  Activation relu1_;
  MaxPool2D pool1_;
  Conv2D conv2_;
  Activation relu2_;
  MaxPool2D pool2_;
  Conv2D conv3_;
  Activation relu3_;
  MaxPool2D pool3_;
  BiLSTM bilstm_;
  Linear fc_;
};

/**
 * @struct SynthTextGenerator
 * @brief Lineas de texto sinteticas para ejercitar el CRNN sin datos reales.
 *
 * No hay tipografias aqui: cada clase del vocabulario se dibuja como un patron
 * binario de 8x4 celdas derivado de su indice, siempre el mismo. Son glifos
 * inventados, y conviene decirlo, pero cumplen lo que el modelo necesita
 * aprender —columnas distinguibles entre si y estables entre muestras—, de modo
 * que la perdida bajando significa algo. El generador anterior devolvia ruido
 * gaussiano con etiquetas `i % 4`: ninguna red podia acertar mas que por azar.
 *
 * Cada caracter ocupa `kCharWidth` columnas, que es exactamente lo que la red
 * reduce el ancho, asi que sale un paso de la secuencia por caracter y la
 * supervision es directa: `targets[b][t]` es la clase del caracter `t`.
 */
struct SynthTextGenerator {
  /** @brief Columnas por caracter; coincide con la reduccion de ancho del CRNN. */
  static constexpr int kCharWidth = CRNNModel::kWidthReduction;

  /**
   * @brief Genera un lote de imagenes `[B, 1, 32, len*4]` y objetivos `[B, len]`.
   *
   * `seed` hace el lote reproducible sin tocar el generador global, que lo usan
   * las inicializaciones de pesos.
   */
  static void Generate(Tensor& images, Tensor& targets, int batch, int word_len,
                       int vocab_size, uint32_t seed = 1234, float noise = 0.05f) {
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

 private:
  /** @brief Patron de 32 bits, estable y distinto para cada clase. */
  static uint32_t Glyph(int cls) {
    // Mezcla entera de Knuth: reparte clases consecutivas en patrones que no se
    // parecen, que es lo unico que se le pide.
    uint32_t h = static_cast<uint32_t>(cls + 1) * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    // Al menos un bit encendido: una clase en blanco seria indistinguible del fondo.
    return h | (1u << (static_cast<uint32_t>(cls) % 32u));
  }
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_MODELS_OCR_H_
