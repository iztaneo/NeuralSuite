// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de image/enderezar.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "tensor.h"
#include "image/enderezar.h"

namespace neuralsuite {
namespace image {

double NitidezProyeccion(const std::vector<uint8_t>& tinta, int ancho, int alto, double radianes) {
  const double sen = std::sin(radianes);
  std::vector<double> filas(static_cast<size_t>(alto), 0.0);
  const double centro_x = ancho * 0.5, centro_y = alto * 0.5;

  for (int y = 0; y < alto; ++y) {
    const uint8_t* linea = tinta.data() + static_cast<size_t>(y) * ancho;
    for (int x = 0; x < ancho; ++x) {
      if (!linea[x]) continue;
      // Solo interesa la coordenada vertical tras el giro.
      const double fy = centro_y + (y - centro_y) - (x - centro_x) * sen;
      const int destino = static_cast<int>(fy + 0.5);
      if (destino >= 0 && destino < alto) filas[destino] += 1.0;
    }
  }

  double suma = 0.0;
  for (int y = 1; y < alto; ++y) {
    const double d = filas[y] - filas[y - 1];
    suma += d * d;
  }
  return suma;
}

double EstimarInclinacion(const Bitmap& imagen, double rango_grados, double paso_grueso, double paso_fino, Binarizacion metodo) {
  if (imagen.Empty()) return 0.0;

  std::vector<float> gris;
  ToGrayscale(imagen, &gris);
  const int ancho = imagen.width, alto = imagen.height;
  // Se binariza una vez y se reutiliza en todas las pruebas: lo que cambia
  // entre angulos es donde cae cada pixel, no cual es tinta.
  const std::vector<uint8_t> tinta = Binarizar(gris, ancho, alto, metodo);

  const auto buscar = [&](double desde, double hasta, double paso) {
    const int n = static_cast<int>(std::lround((hasta - desde) / paso)) + 1;
    std::vector<double> puntuacion(static_cast<size_t>(std::max(1, n)), 0.0);
    parallel::ParallelFor(n, /*min_per_thread=*/1, [&](int a, int b) {
      for (int i = a; i < b; ++i) {
        puntuacion[i] = NitidezProyeccion(tinta, ancho, alto,
                                          (desde + i * paso) * kPi / 180.0);
      }
    });
    int mejor = 0;
    for (int i = 1; i < n; ++i) {
      if (puntuacion[i] > puntuacion[mejor]) mejor = i;
    }
    return desde + mejor * paso;
  };

  const double grueso = buscar(-rango_grados, rango_grados, paso_grueso);
  return buscar(grueso - paso_grueso, grueso + paso_grueso, paso_fino);
}

Bitmap Girar(const Bitmap& imagen, double grados) {
  Bitmap salida;
  salida.width = imagen.width;
  salida.height = imagen.height;
  salida.channels = imagen.channels;
  const int C = imagen.channels;

  // Nivel del fondo: la mediana del borde, por canal.
  std::vector<uint8_t> fondo(static_cast<size_t>(C), 255);
  for (int c = 0; c < C; ++c) {
    std::vector<uint8_t> muestras;
    for (int x = 0; x < imagen.width; ++x) {
      muestras.push_back(imagen.pixels[static_cast<size_t>(x) * C + c]);
      muestras.push_back(
          imagen.pixels[(static_cast<size_t>(imagen.height - 1) * imagen.width + x) * C + c]);
    }
    if (!muestras.empty()) {
      std::nth_element(muestras.begin(), muestras.begin() + muestras.size() / 2, muestras.end());
      fondo[c] = muestras[muestras.size() / 2];
    }
  }

  salida.pixels.assign(static_cast<size_t>(salida.width) * salida.height * C, 0);
  const double radianes = grados * kPi / 180.0;
  const double sen = std::sin(radianes), cos_ = std::cos(radianes);
  const double cx = imagen.width * 0.5, cy = imagen.height * 0.5;

  parallel::ParallelFor(salida.height, /*min_per_thread=*/8, [&](int desde, int hasta) {
    for (int y = desde; y < hasta; ++y) {
      for (int x = 0; x < salida.width; ++x) {
        // Giro inverso: de donde viene este pixel del destino.
        const double dx = x - cx, dy = y - cy;
        const double ox = cx + dx * cos_ + dy * sen;
        const double oy = cy - dx * sen + dy * cos_;

        uint8_t* destino = salida.pixels.data() + (static_cast<size_t>(y) * salida.width + x) * C;
        if (ox < 0 || oy < 0 || ox > imagen.width - 1 || oy > imagen.height - 1) {
          for (int c = 0; c < C; ++c) destino[c] = fondo[c];
          continue;
        }
        const int x0 = static_cast<int>(ox), y0 = static_cast<int>(oy);
        const int x1 = std::min(x0 + 1, imagen.width - 1);
        const int y1 = std::min(y0 + 1, imagen.height - 1);
        const double wx = ox - x0, wy = oy - y0;

        for (int c = 0; c < C; ++c) {
          const double a = imagen.pixels[(static_cast<size_t>(y0) * imagen.width + x0) * C + c];
          const double b = imagen.pixels[(static_cast<size_t>(y0) * imagen.width + x1) * C + c];
          const double d = imagen.pixels[(static_cast<size_t>(y1) * imagen.width + x0) * C + c];
          const double e = imagen.pixels[(static_cast<size_t>(y1) * imagen.width + x1) * C + c];
          const double v = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + d * (1 - wx) * wy +
                           e * wx * wy;
          destino[c] = static_cast<uint8_t>(std::lround(std::max(0.0, std::min(255.0, v))));
        }
      }
    }
  });
  return salida;
}

Bitmap Enderezar(const Bitmap& imagen, double* grados_aplicados, double minimo_grados) {
  const double grados = EstimarInclinacion(imagen);
  if (grados_aplicados) *grados_aplicados = grados;
  if (std::abs(grados) < minimo_grados) return imagen;
  return Girar(imagen, -grados);
}

}  // namespace image
}  // namespace neuralsuite
