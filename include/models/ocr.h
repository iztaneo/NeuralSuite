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

  /**
   * @brief Vocabulario por defecto: 62 simbolos, el espacio y el blanco.
   *
   * La ultima clase es el blanco, que marca «aqui no hay letra» y el
   * decodificado descarta. El espacio entre palabras es una clase aparte y si
   * se conserva. Antes eran la misma, y por eso el modelo no podia leer una
   * linea de varias palabras aunque acertara todos los caracteres: el
   * decodificado se comia los espacios junto con el relleno.
   */
  /**
   * @brief Parte una cadena UTF-8 en simbolos.
   *
   * Cada simbolo del vocabulario es una clase, pero en UTF-8 un simbolo no es
   * un byte: `a` ocupa uno, `á` y `ñ` dos, la raya `—` tres. Indexar por byte
   * —que es lo que hacia este codigo con `std::vector<char>`— convertia la
   * clase 26 en medio simbolo en cuanto el vocabulario dejaba de ser ASCII.
   *
   * La longitud de cada secuencia esta en su primer byte; los que empiezan por
   * `10` son continuacion y no abren simbolo.
   */
  static std::vector<std::string> PartirUtf8(const std::string& texto);

  /**
   * @brief Vocabulario por defecto: letras, digitos, acentos y puntuacion.
   *
   * Los acentos y los signos no estan por completitud: medido sobre la pagina
   * de la Iliada, eran el 4.0% de sus caracteres y el modelo no podia emitir
   * ninguno. `ó` salia como `6`, `ú` como `i`, la raya como `rzu`, y las comas
   * y los puntos desaparecian.
   *
   * La clase de blanco no aparece aqui ni en el archivo de vocabulario: es
   * siempre la ultima, `vocab.size()`. Guardarla dentro obligaria a representar
   * un caracter que no se imprime, y un NUL dentro de un archivo de texto es
   * una fuente de sorpresas.
   */
  static std::vector<std::string> DefaultVocab();

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
  Tensor Forward(const Tensor& images) override;

  /** @brief `[B, T, num_classes]` -> `[B, C, 32, W]`. */
  Tensor Backward(const Tensor& dout) override;

  /**
   * @brief Convierte los logits en una cadena por cada imagen del lote.
   *
   * Colapsa repeticiones consecutivas del mismo simbolo y descarta el relleno.
   * Es un decodificado voraz: no busca la secuencia mas probable en conjunto,
   * solo el mejor simbolo en cada paso.
   *
   * `clase_blanco` dice cual es la clase de relleno. Con -1 se descarta el
   * espacio, que es lo que hacia la implementacion de referencia y sigue
   * valiendo para un vocabulario sin blanco. Con un vocabulario que lo tenga,
   * hay que pasarlo: si no, los espacios entre palabras sobreviven —bien— pero
   * el relleno tambien, y la salida se llena de basura.
   */
  [[nodiscard]] std::vector<std::string> DecodeBatch(const Tensor& logits, const std::vector<std::string>& vocab, int clase_blanco = -1) const;

  /** @brief La primera imagen del lote, para el caso de una sola linea. */
  [[nodiscard]] std::string DecodeWord(const Tensor& logits, const std::vector<std::string>& vocab, int clase_blanco = -1) const;

  bool Save(const std::string& path);

  bool Load(const std::string& path);

  /** @brief Descripcion que viaja dentro del archivo de pesos. */
  [[nodiscard]] std::map<std::string, std::string> ArchitectureMetadata() const;

  [[nodiscard]] int NumClasses() const { return num_classes_; }
  /** @brief Cuantas predicciones devuelve la red para un ancho dado. */
  [[nodiscard]] static int TimestepsFor(int width) { return width / kWidthReduction; }

 private:
  /** @brief Canales que salen del extractor convolucional. */
  static constexpr int kSeqFeatures = 64;

  /** @brief `[B, C, 1, T]` -> `[T, B, C]`. */
  static Tensor MapToSequence(const Tensor& h, int batch, int channels, int steps);

  /** @brief La inversa de MapToSequence, para el gradiente. */
  static Tensor SequenceToMap(const Tensor& seq, int batch, int channels, int steps);

  /**
   * @brief Intercambia los dos primeros ejes de un `[d0, d1, k]`.
   *
   * Sirve en los dos sentidos —de `[T, B, K]` a `[B, T, K]` y al reves— porque
   * intercambiar dos ejes es su propia inversa: basta pasar los tamanos en el
   * orden que tiene el tensor de entrada. Estaban escritas como dos funciones
   * distintas, y no lo eran: calculaban exactamente lo mismo, de modo que
   * confundir una con otra no daba error pero invitaba a creer que si lo daria.
   */
  static Tensor SwapFirstTwoAxes(const Tensor& x, int d0, int d1, int k);

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
  static void Generate(Tensor& images, Tensor& targets, int batch, int word_len, int vocab_size, uint32_t seed = 1234, float noise = 0.05f);

 private:
  /** @brief Patron de 32 bits, estable y distinto para cada clase. */
  static uint32_t Glyph(int cls);
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_MODELS_OCR_H_
