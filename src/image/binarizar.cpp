// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/binarizar.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "image/binarizar.h"

namespace neuralsuite {
namespace image {

int UmbralOtsu(const std::vector<float>& gris) {
  int histograma[256] = {0};
  for (float v : gris) {
    const int nivel = static_cast<int>(v * 255.0f + 0.5f);
    histograma[std::max(0, std::min(255, nivel))]++;
  }
  const double total = static_cast<double>(gris.size());
  double suma_total = 0.0;
  for (int i = 0; i < 256; ++i) suma_total += i * histograma[i];

  double suma_fondo = 0.0, peso_fondo = 0.0, mejor_varianza = -1.0;
  int mejor = 128;
  for (int t = 0; t < 256; ++t) {
    peso_fondo += histograma[t];
    if (peso_fondo == 0.0) continue;
    const double peso_frente = total - peso_fondo;
    if (peso_frente == 0.0) break;
    suma_fondo += t * histograma[t];
    const double media_fondo = suma_fondo / peso_fondo;
    const double media_frente = (suma_total - suma_fondo) / peso_frente;
    const double diferencia = media_fondo - media_frente;
    const double varianza = peso_fondo * peso_frente * diferencia * diferencia;
    if (varianza > mejor_varianza) {
      mejor_varianza = varianza;
      mejor = t;
    }
  }
  return mejor;
}

std::vector<uint8_t> BinarizarGlobal(const std::vector<float>& gris) {
  const int umbral = UmbralOtsu(gris);
  std::vector<uint8_t> salida(gris.size(), 0);
  for (size_t i = 0; i < gris.size(); ++i) {
    salida[i] = static_cast<int>(gris[i] * 255.0f + 0.5f) <= umbral ? 1 : 0;
  }
  return salida;
}

int VentanaSauvola(int ancho, int alto) {
  const int menor = std::min(ancho, alto);
  int v = std::max(15, std::min(51, menor / 12));
  if (v % 2 == 0) ++v;
  return v;
}

std::vector<uint8_t> BinarizarSauvola(const std::vector<float>& gris, int ancho, int alto, int ventana, double k, double R) {
  if (ventana <= 0) ventana = VentanaSauvola(ancho, alto);
  const int radio = ventana / 2;

  // Imagenes integrales de los valores y de sus cuadrados, con una fila y una
  // columna de ceros al principio para no tener que tratar los bordes aparte.
  const size_t W = static_cast<size_t>(ancho) + 1, H = static_cast<size_t>(alto) + 1;
  std::vector<double> suma(W * H, 0.0), suma2(W * H, 0.0);
  for (int y = 0; y < alto; ++y) {
    double fila = 0.0, fila2 = 0.0;
    for (int x = 0; x < ancho; ++x) {
      const double v = gris[static_cast<size_t>(y) * ancho + x] * 255.0;
      fila += v;
      fila2 += v * v;
      suma[(y + 1) * W + (x + 1)] = suma[static_cast<size_t>(y) * W + (x + 1)] + fila;
      suma2[(y + 1) * W + (x + 1)] = suma2[static_cast<size_t>(y) * W + (x + 1)] + fila2;
    }
  }

  std::vector<uint8_t> salida(gris.size(), 0);
  parallel::ParallelFor(alto, /*min_per_thread=*/8, [&](int desde, int hasta) {
    for (int y = desde; y < hasta; ++y) {
      const int y0 = std::max(0, y - radio), y1 = std::min(alto - 1, y + radio);
      for (int x = 0; x < ancho; ++x) {
        const int x0 = std::max(0, x - radio), x1 = std::min(ancho - 1, x + radio);
        const double n = static_cast<double>(y1 - y0 + 1) * (x1 - x0 + 1);

        // Cuatro lecturas por rectangulo, sea del tamano que sea.
        const auto rect = [&](const std::vector<double>& tabla) {
          return tabla[static_cast<size_t>(y1 + 1) * W + (x1 + 1)] -
                 tabla[static_cast<size_t>(y0) * W + (x1 + 1)] -
                 tabla[static_cast<size_t>(y1 + 1) * W + x0] +
                 tabla[static_cast<size_t>(y0) * W + x0];
        };
        const double media = rect(suma) / n;
        // La varianza puede salir levemente negativa por redondeo cuando la
        // vecindad es plana; se recorta antes de la raiz.
        const double varianza = std::max(0.0, rect(suma2) / n - media * media);
        const double desviacion = std::sqrt(varianza);

        const double umbral = media * (1.0 + k * (desviacion / R - 1.0));
        const double valor = gris[static_cast<size_t>(y) * ancho + x] * 255.0;
        salida[static_cast<size_t>(y) * ancho + x] = (valor <= umbral) ? 1 : 0;
      }
    }
  });
  return salida;
}

double NitidezDeFilas(const std::vector<uint8_t>& marca, int ancho, int alto) {
  std::vector<double> filas(static_cast<size_t>(alto), 0.0);
  double tinta = 0.0;
  for (int y = 0; y < alto; ++y) {
    double suma = 0.0;
    for (int x = 0; x < ancho; ++x) suma += marca[static_cast<size_t>(y) * ancho + x];
    filas[y] = suma;
    tinta += suma;
  }
  if (tinta <= 0.0) return 0.0;
  double nitidez = 0.0;
  for (int y = 1; y < alto; ++y) {
    const double d = filas[y] - filas[y - 1];
    nitidez += d * d;
  }
  return nitidez / tinta;
}

std::vector<uint8_t> Binarizar(const std::vector<float>& gris, int ancho, int alto, Binarizacion metodo) {
  if (metodo == Binarizacion::kGlobal) return BinarizarGlobal(gris);
  if (metodo == Binarizacion::kAdaptativa) return BinarizarSauvola(gris, ancho, alto);

  std::vector<uint8_t> global = BinarizarGlobal(gris);
  std::vector<uint8_t> adaptativa = BinarizarSauvola(gris, ancho, alto);
  return NitidezDeFilas(adaptativa, ancho, alto) > NitidezDeFilas(global, ancho, alto)
             ? std::move(adaptativa)
             : std::move(global);
}

}  // namespace image
}  // namespace neuralsuite
