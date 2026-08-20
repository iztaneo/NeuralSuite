// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file train_ocr.cpp
 * @brief Entrena el CRNN a leer texto renderizado con tipografias reales.
 *
 * Lee un corpus generado por tools/ocr/generar_dataset.py: imagenes PNG y un
 * archivo de etiquetas. Las imagenes se decodifican con include/image/, de modo
 * que el entrenamiento no depende de Python ni de ninguna biblioteca externa.
 *
 * La supervision es paso a paso. El CRNN devuelve una prediccion por cada
 * cuatro columnas de la imagen, y el generador sabe que letra cae en cada una
 * porque es quien las dibujo, asi que cada paso tiene su etiqueta y basta
 * `CrossEntropyLoss`. Es la alternativa a CTC cuando la alineacion se conoce, y
 * evita anadir una perdida nueva sin verificar. Lo que se pierde: no se puede
 * entrenar con datos ajenos ya etiquetados, donde la alineacion se desconoce.
 *
 * Uso:
 *   ./train_ocr --datos /tmp/ocr_datos --epocas 20 --lote 8
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "neuralsuite.h"

using namespace neuralsuite;

namespace {

/** @brief Una linea del corpus: la imagen ya cargada y sus etiquetas. */
struct Muestra {
  std::vector<float> pixeles;   // [alto * ancho], tinta clara sobre fondo oscuro
  std::vector<int> por_paso;    // clase de cada paso de la secuencia
  std::string texto;
};

/** @brief Carga una particion del corpus. Devuelve false si falta algo. */
bool CargarParticion(const std::string& raiz, const std::string& nombre, int alto, int ancho,
                     std::vector<Muestra>* salida, std::string* error) {
  const std::string lista = raiz + "/" + nombre + ".txt";
  std::ifstream fichero(lista);
  if (!fichero) {
    *error = "no se pudo abrir " + lista;
    return false;
  }

  std::string linea;
  while (std::getline(fichero, linea)) {
    if (linea.empty()) continue;
    const size_t tab1 = linea.find('\t');
    const size_t tab2 = linea.find('\t', tab1 + 1);
    if (tab1 == std::string::npos || tab2 == std::string::npos) {
      *error = "linea con formato inesperado en " + lista + ": " + linea;
      return false;
    }

    Muestra muestra;
    muestra.texto = linea.substr(tab1 + 1, tab2 - tab1 - 1);
    std::istringstream pasos(linea.substr(tab2 + 1));
    int clase;
    while (pasos >> clase) muestra.por_paso.push_back(clase);

    const std::string ruta = raiz + "/" + nombre + "/" + linea.substr(0, tab1);
    image::Bitmap mapa;
    if (!image::Load(ruta, &mapa, error)) return false;
    if (mapa.width != ancho || mapa.height != alto) {
      *error = ruta + " mide " + std::to_string(mapa.width) + "x" +
               std::to_string(mapa.height) + " y se esperaba " + std::to_string(ancho) + "x" +
               std::to_string(alto);
      return false;
    }

    std::vector<float> gris;
    image::ToGrayscale(mapa, &gris);
    // Se invierte: el corpus trae tinta oscura sobre papel claro, y a la red le
    // conviene que el trazo sea la senal fuerte y el fondo el cero.
    muestra.pixeles.resize(gris.size());
    for (size_t i = 0; i < gris.size(); ++i) muestra.pixeles[i] = 1.0f - gris[i];

    salida->push_back(std::move(muestra));
  }
  return true;
}

/** @brief Colapsa repeticiones y descarta espacios, igual que DecodeBatch. */
std::string Colapsar(const std::vector<int>& clases, const std::string& vocabulario) {
  std::string salida;
  int previo = -1;
  for (int clase : clases) {
    if (clase != previo) {
      if (clase >= 0 && clase < static_cast<int>(vocabulario.size()) &&
          vocabulario[clase] != ' ') {
        salida += vocabulario[clase];
      }
      previo = clase;
    }
  }
  return salida;
}

/** @brief Distancia de edicion, para medir cuanto se acerca sin acertar del todo. */
int DistanciaEdicion(const std::string& a, const std::string& b) {
  std::vector<int> anterior(b.size() + 1), actual(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) anterior[j] = static_cast<int>(j);
  for (size_t i = 1; i <= a.size(); ++i) {
    actual[0] = static_cast<int>(i);
    for (size_t j = 1; j <= b.size(); ++j) {
      const int coste = (a[i - 1] == b[j - 1]) ? 0 : 1;
      actual[j] = std::min({anterior[j] + 1, actual[j - 1] + 1, anterior[j - 1] + coste});
    }
    anterior = actual;
  }
  return anterior[b.size()];
}

struct Evaluacion {
  double acierto_por_paso = 0.0;
  double acierto_palabra = 0.0;
  double error_caracter = 0.0;
};

Evaluacion Evaluar(CRNNModel& modelo, const std::vector<Muestra>& datos, int alto, int ancho,
                   int lote, const std::string& vocabulario) {
  const std::vector<char> vocab(vocabulario.begin(), vocabulario.end());
  long pasos_ok = 0, pasos_total = 0, palabras_ok = 0;
  long ediciones = 0, caracteres = 0;

  for (size_t inicio = 0; inicio < datos.size(); inicio += lote) {
    const int n = static_cast<int>(std::min<size_t>(lote, datos.size() - inicio));
    Tensor x({n, 1, alto, ancho});
    for (int b = 0; b < n; ++b) {
      std::copy(datos[inicio + b].pixeles.begin(), datos[inicio + b].pixeles.end(),
                x.Data() + static_cast<size_t>(b) * alto * ancho);
    }
    const Tensor logits = modelo.Forward(x);
    const std::vector<std::string> leidas = modelo.DecodeBatch(logits, vocab);

    const int T = logits.Shape()[1], C = logits.Shape()[2];
    for (int b = 0; b < n; ++b) {
      const Muestra& m = datos[inicio + b];
      for (int t = 0; t < T && t < static_cast<int>(m.por_paso.size()); ++t) {
        int mejor = 0;
        float valor = logits[(static_cast<size_t>(b) * T + t) * C];
        for (int c = 1; c < C; ++c) {
          const float v = logits[(static_cast<size_t>(b) * T + t) * C + c];
          if (v > valor) { valor = v; mejor = c; }
        }
        pasos_ok += (mejor == m.por_paso[t]) ? 1 : 0;
        ++pasos_total;
      }
      if (leidas[b] == m.texto) ++palabras_ok;
      ediciones += DistanciaEdicion(leidas[b], m.texto);
      caracteres += static_cast<long>(m.texto.size());
    }
  }

  Evaluacion e;
  e.acierto_por_paso = pasos_total ? static_cast<double>(pasos_ok) / pasos_total : 0.0;
  e.acierto_palabra = datos.empty() ? 0.0 : static_cast<double>(palabras_ok) / datos.size();
  e.error_caracter = caracteres ? static_cast<double>(ediciones) / caracteres : 0.0;
  return e;
}

}  // namespace

int main(int argc, char** argv) {
  std::string raiz = "/tmp/ocr_datos";
  std::string salida;
  int epocas = 20, lote = 8, oculto = 64, alto = 32, ancho = 128, limite = 0;
  float lr = 1e-3f;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--datos" && i + 1 < argc) raiz = argv[++i];
    else if (arg == "--salida" && i + 1 < argc) salida = argv[++i];
    else if (arg == "--epocas" && i + 1 < argc) epocas = std::stoi(argv[++i]);
    else if (arg == "--lote" && i + 1 < argc) lote = std::stoi(argv[++i]);
    else if (arg == "--oculto" && i + 1 < argc) oculto = std::stoi(argv[++i]);
    else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
    else if (arg == "--limite" && i + 1 < argc) limite = std::stoi(argv[++i]);
    else if (arg == "--help") {
      std::printf("Uso: train_ocr [--datos ruta] [--salida ruta] [--epocas n] [--lote n]\n"
                  "               [--oculto n] [--lr f] [--limite n]\n\n"
                  "  --limite n   usa solo las n primeras imagenes de entrenamiento,\n"
                  "               para una pasada corta que compruebe la tuberia.\n");
      return 0;
    }
  }
  if (salida.empty()) salida = ReleasePath("ocr_texto.ns");

  std::string vocabulario;
  {
    std::ifstream fichero(raiz + "/vocab.txt");
    if (!fichero) {
      std::cerr << "ERROR: no se pudo abrir " << raiz << "/vocab.txt\n";
      return 1;
    }
    std::getline(fichero, vocabulario);
  }

  std::cout << "============================================================\n";
  std::cout << "Entrenamiento de OCR sobre texto renderizado\n";
  std::cout << "============================================================\n" << std::flush;

  std::string error;
  std::vector<Muestra> entrenamiento, validacion;
  std::printf("decodificando las imagenes del corpus...\n");
  std::fflush(stdout);
  const auto t_carga = std::chrono::steady_clock::now();
  if (!CargarParticion(raiz, "train", alto, ancho, &entrenamiento, &error) ||
      !CargarParticion(raiz, "val", alto, ancho, &validacion, &error)) {
    std::cerr << "ERROR al cargar el corpus: " << error << "\n";
    return 1;
  }
  if (limite > 0 && limite < static_cast<int>(entrenamiento.size())) {
    entrenamiento.resize(limite);
  }
  const double ms_carga = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_carga).count();

  const int pasos = CRNNModel::TimestepsFor(ancho);
  std::printf("corpus      : %zu entrenamiento, %zu validacion  (%.1f s en decodificar)\n",
              entrenamiento.size(), validacion.size(), ms_carga / 1000.0);
  std::printf("vocabulario : %zu simbolos\n", vocabulario.size());
  std::printf("imagen      : %dx%d  ->  %d pasos por linea\n", ancho, alto, pasos);
  std::printf("modelo      : oculto=%d, lote=%d, lr=%g, %d epocas\n\n", oculto, lote, lr, epocas);

  ManualSeed(1234);
  CRNNModel modelo(1, oculto, static_cast<int>(vocabulario.size()));
  AdamW optimizador(modelo.Parameters(), lr);
  CrossEntropyLoss criterio;

  std::vector<int> orden(entrenamiento.size());
  std::iota(orden.begin(), orden.end(), 0);
  std::mt19937 generador(7);

  double mejor_palabra = -1.0;
  std::printf("%-7s %12s %14s %14s %12s %10s\n",
              "epoca", "perdida", "aciertos/paso", "palabras", "error car.", "min");
  std::printf("---------------------------------------------------------------------------\n");

  const auto t_inicio = std::chrono::steady_clock::now();
  for (int epoca = 1; epoca <= epocas; ++epoca) {
    // Barajar cada epoca: si el orden fuera fijo, el optimizador veria siempre
    // la misma secuencia de lotes y eso introduce una periodicidad que no esta
    // en los datos.
    std::shuffle(orden.begin(), orden.end(), generador);

    double perdida_total = 0.0;
    int lotes = 0;
    // Cada epoca son varios minutos. Sin senal dentro de ella, seguir el
    // entrenamiento con `tail -f` es mirar una pantalla quieta y no poder
    // distinguir «va lento» de «se colgo».
    const int total_lotes = static_cast<int>(orden.size() / lote);
    const int cada = std::max(1, total_lotes / 10);
    const auto t_epoca = std::chrono::steady_clock::now();

    for (size_t inicio = 0; inicio + lote <= orden.size(); inicio += lote) {
      optimizador.ZeroGrad();

      Tensor x({lote, 1, alto, ancho});
      Tensor objetivos({lote * pasos});
      for (int b = 0; b < lote; ++b) {
        const Muestra& m = entrenamiento[orden[inicio + b]];
        std::copy(m.pixeles.begin(), m.pixeles.end(),
                  x.Data() + static_cast<size_t>(b) * alto * ancho);
        for (int t = 0; t < pasos; ++t) {
          objetivos[static_cast<size_t>(b) * pasos + t] =
              static_cast<float>(t < static_cast<int>(m.por_paso.size()) ? m.por_paso[t] : 0);
        }
      }

      Tensor logits = modelo.Forward(x);
      Tensor plano = logits;
      plano.Reshape({lote * pasos, static_cast<int>(vocabulario.size())});
      perdida_total += criterio.Forward(plano, objetivos);
      ++lotes;

      Tensor d = criterio.Backward();
      d.Reshape(logits.Shape());
      modelo.Backward(d);
      optimizador.Step();

      if (lotes % cada == 0 || lotes == total_lotes) {
        const double seg = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_epoca).count();
        const double restante = lotes ? seg * (total_lotes - lotes) / lotes : 0.0;
        std::printf("  epoca %d: %d/%d lotes  perdida %.4f  %.0f img/s  quedan %.0f s\n",
                    epoca, lotes, total_lotes, perdida_total / lotes,
                    lotes * lote / std::max(seg, 1e-9), restante);
        std::fflush(stdout);
      }
    }

    const Evaluacion e = Evaluar(modelo, validacion, alto, ancho, lote, vocabulario);
    const double minutos = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_inicio).count() / 60.0;
    std::printf("%-7d %12.5f %13.1f%% %13.1f%% %11.3f %10.1f\n", epoca,
                lotes ? perdida_total / lotes : 0.0, 100 * e.acierto_por_paso,
                100 * e.acierto_palabra, e.error_caracter, minutos);
    std::fflush(stdout);

    if (e.acierto_palabra > mejor_palabra) {
      mejor_palabra = e.acierto_palabra;
      if (!modelo.Save(salida)) {
        std::cerr << "AVISO: no se pudieron guardar los pesos en " << salida << "\n";
      }
    }
  }

  std::printf("\nMejor acierto por palabra en validacion: %.1f%%\n", 100 * mejor_palabra);
  std::printf("Pesos guardados en: %s\n", salida.c_str());

  // Unas cuantas lecturas concretas, que dicen mas que un porcentaje.
  if (!validacion.empty()) {
    modelo.Load(salida);
    const std::vector<char> vocab(vocabulario.begin(), vocabulario.end());
    const int n = static_cast<int>(std::min<size_t>(8, validacion.size()));
    Tensor x({n, 1, alto, ancho});
    for (int b = 0; b < n; ++b) {
      std::copy(validacion[b].pixeles.begin(), validacion[b].pixeles.end(),
                x.Data() + static_cast<size_t>(b) * alto * ancho);
    }
    const std::vector<std::string> leidas = modelo.DecodeBatch(modelo.Forward(x), vocab);
    std::printf("\nLecturas de validacion:\n");
    for (int b = 0; b < n; ++b) {
      std::printf("  %s  leido '%s'   esperado '%s'\n",
                  leidas[b] == validacion[b].texto ? "OK  " : "FALLA",
                  leidas[b].c_str(), validacion[b].texto.c_str());
    }
  }
  std::printf("============================================================\n");
  return 0;
}
