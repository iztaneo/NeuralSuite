// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file eval_llm.cpp
 * @brief Mide la pérdida y la perplejidad de un modelo sobre texto que no vio.
 *
 * El entrenamiento sólo informa de la pérdida sobre los lotes que va usando, y
 * eso no dice si el modelo aprendió el idioma o se aprendió el corpus. La
 * diferencia sólo aparece midiendo sobre texto apartado, y en este proyecto ya
 * costó caro no hacerlo: en OCR las métricas de validación mejoraban mientras el
 * rendimiento sobre una página real empeoraba, porque validación y entrenamiento
 * salían del mismo generador.
 *
 * Por eso `corpus/es/` tiene tres particiones y no dos, y la tercera es un autor
 * entero que nunca aparece en entrenamiento.
 *
 * Uso:
 *   ./bin/eval_llm --model release/es_base.bin --vocab release/es_base_vocab.txt
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "neuralsuite.h"

namespace {

std::string LeerArchivo(const std::string& ruta) {
  std::ifstream f(ruta);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string modelo_ruta = "release/es_base.bin";
  std::string vocab_ruta = "release/es_base_vocab.txt";
  std::string corpus = "corpus/es";
  int block_size = 128, n_layer = 4, n_head = 4, n_embd = 128;
  int lote = 8, max_lotes = 30;
  size_t max_caracteres = 400000;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "Uso: " << argv[0] << " [OPCIONES]\n\n"
                << "  --model <path>      Pesos (default: release/es_base.bin)\n"
                << "  --vocab <path>      Vocabulario (default: release/es_base_vocab.txt)\n"
                << "  --corpus <dir>      Directorio con train/val/test.txt (default: corpus/es)\n"
                << "  --block_size <int>  (default: 128)\n"
                << "  --n_layer <int>     (default: 4)\n"
                << "  --n_head <int>      (default: 4)\n"
                << "  --n_embd <int>      (default: 128)\n";
      return 0;
    } else if (a == "--model" && i + 1 < argc) { modelo_ruta = argv[++i];
    } else if (a == "--vocab" && i + 1 < argc) { vocab_ruta = argv[++i];
    } else if (a == "--corpus" && i + 1 < argc) { corpus = argv[++i];
    } else if (a == "--block_size" && i + 1 < argc) { block_size = std::stoi(argv[++i]);
    } else if (a == "--n_layer" && i + 1 < argc) { n_layer = std::stoi(argv[++i]);
    } else if (a == "--n_head" && i + 1 < argc) { n_head = std::stoi(argv[++i]);
    } else if (a == "--n_embd" && i + 1 < argc) { n_embd = std::stoi(argv[++i]);
    }
  }

  neuralsuite::CharTokenizer tok;
  tok.Load(vocab_ruta);
  if (tok.VocabSize() <= 0) {
    std::cerr << "ERROR: no se pudo cargar el vocabulario '" << vocab_ruta << "'.\n";
    return 1;
  }

  neuralsuite::GPTConfig cfg;
  cfg.vocab_size = tok.VocabSize();
  cfg.block_size = block_size;
  cfg.n_layer = n_layer;
  cfg.n_head = n_head;
  cfg.n_embd = n_embd;
  neuralsuite::GPTModel modelo(cfg);

  // Se aborta si los pesos no cargan, en vez de medir pesos aleatorios y dar una
  // cifra. Ya pasó una vez en OCR: un `2>/dev/null` escondió el fallo de carga y
  // estuve a punto de reportar como resultado la medida de una red sin entrenar.
  if (!modelo.LoadWeights(modelo_ruta)) {
    std::cerr << "ERROR: no se cargaron los pesos de '" << modelo_ruta
              << "'. No se mide nada: una cifra sobre pesos aleatorios es peor "
                 "que ninguna cifra.\n";
    return 1;
  }
  std::cout << "✅ Pesos cargados. Vocabulario: " << cfg.vocab_size << " símbolos.\n\n";

  neuralsuite::CrossEntropyLoss criterio;
  std::printf("  %-8s %12s %10s %13s\n", "", "caracteres", "pérdida", "perplejidad");

  for (const char* nombre : {"train", "val", "test"}) {
    std::string txt = LeerArchivo(corpus + "/" + nombre + ".txt");
    if (txt.empty()) {
      std::cerr << "ERROR: '" << corpus << "/" << nombre << ".txt' está vacío o no existe.\n";
      return 1;
    }
    if (txt.size() > max_caracteres) txt = txt.substr(0, max_caracteres);
    const std::vector<int> ids = tok.Encode(txt);

    // Bloques disjuntos: sin solape no se cuenta dos veces el mismo texto.
    const size_t paso = static_cast<size_t>(lote) * block_size;
    double suma = 0.0;
    int lotes = 0;
    for (size_t o = 0; o + paso + 1 < ids.size() && lotes < max_lotes; o += paso) {
      neuralsuite::Tensor idx({lote, block_size}), obj({lote * block_size});
      for (size_t i = 0; i < paso; ++i) {
        idx[i] = static_cast<float>(ids[o + i]);
        obj[i] = static_cast<float>(ids[o + i + 1]);
      }
      const neuralsuite::Tensor logits = modelo.Forward(idx);
      // La pérdida espera los logits en 2D [N, V]; pasárselos en 3D da una cifra
      // absurda —medido: 12.13, peor que el azar— sin ningún aviso.
      neuralsuite::Tensor logits_2d({lote * block_size, cfg.vocab_size});
      std::memcpy(logits_2d.Data(), logits.Data(), logits.TotalSize() * sizeof(float));
      suma += criterio.Forward(logits_2d, obj);
      ++lotes;
    }
    if (lotes == 0) {
      std::cerr << "ERROR: '" << nombre << "' no da ni un lote completo.\n";
      return 1;
    }
    const double L = suma / lotes;
    std::printf("  %-8s %12zu %10.4f %13.2f\n", nombre, txt.size(), L, std::exp(L));
  }

  std::printf(
      "\n  La distancia entre `train` y `test` es lo que dice si aprendió el\n"
      "  idioma o el corpus: `test` es un autor que nunca apareció en\n"
      "  entrenamiento. El azar puro daría perplejidad igual al vocabulario.\n");
  return 0;
}
