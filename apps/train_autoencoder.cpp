// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file train_autoencoder.cpp
 * @brief Entrena el autoencoder convolucional del LDM-1 sobre MNIST.
 *
 * Imagen 28×28 rellenada a 32×32 → Codificador → GaussianaDiagonal → latente
 * 8×8×C → Decodificador → reconstruccion 32×32.
 *
 * La perdida sigue la del autoencoder KL de Stable Diffusion:
 *
 *     L = 1/N · sum (x_rec − x)²  +  peso_kl · KL
 *
 * con el error de reconstruccion SUMADO sobre los pixeles de cada imagen y
 * promediado en el lote, igual que la KL. Si la reconstruccion se promediara
 * por pixel y la KL no, los dos terminos quedarian en escalas distintas y el
 * `peso_kl` del paper (1e-6) no significaria lo mismo aqui. Se usa error
 * cuadratico en vez del L1 + perdida perceptual del paper porque la perceptual
 * necesita una VGG preentrenada, y esa decision esta aplazada.
 *
 * El primer uso es la **puerta de sobreajuste** (`--n_imagenes 16`): si el
 * autoencoder no es capaz de reconstruir casi perfectamente dieciseis digitos,
 * hay un defecto, y ninguna cantidad de datos lo va a arreglar.
 */

#include "neuralsuite.h"
#include "image/png.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace neuralsuite;
using namespace neuralsuite::latent;
using namespace neuralsuite::data;

namespace {

constexpr int kLado = 32;          // MNIST rellenado
constexpr int kPx = kLado * kLado;

/** @brief PSNR en dB para imagenes en [-1, 1], calculado sobre la escala [0, 1]. */
double Psnr(double mse_menos1_1) {
  const double mse01 = mse_menos1_1 / 4.0;   // (a−b)/2 al pasar a [0, 1]
  return (mse01 <= 0.0) ? 99.0 : 10.0 * std::log10(1.0 / mse01);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  std::string dir = "corpus/mnist";
  std::string png = "release/autoencoder_reconstruccion.png";
  int n_imagenes = 16, iteraciones = 600, lote = 16, canales = 32, c_lat = 4, grupos = 8;
  int semilla = 7, reportar_cada = 50;
  float lr = 1e-3f, peso_kl = 1e-6f;

  for (int i = 1; i < argc; ++i) {
    auto sig = [&](const char* q) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Falta el valor de %s\n", q); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--dir")) dir = sig("--dir");
    else if (!std::strcmp(argv[i], "--png")) png = sig("--png");
    else if (!std::strcmp(argv[i], "--n_imagenes")) n_imagenes = std::atoi(sig("--n_imagenes"));
    else if (!std::strcmp(argv[i], "--iteraciones")) iteraciones = std::atoi(sig("--iteraciones"));
    else if (!std::strcmp(argv[i], "--lote")) lote = std::atoi(sig("--lote"));
    else if (!std::strcmp(argv[i], "--canales")) canales = std::atoi(sig("--canales"));
    else if (!std::strcmp(argv[i], "--c_lat")) c_lat = std::atoi(sig("--c_lat"));
    else if (!std::strcmp(argv[i], "--lr")) lr = static_cast<float>(std::atof(sig("--lr")));
    else if (!std::strcmp(argv[i], "--peso_kl")) peso_kl = static_cast<float>(std::atof(sig("--peso_kl")));
    else if (!std::strcmp(argv[i], "--semilla")) semilla = std::atoi(sig("--semilla"));
    else if (!std::strcmp(argv[i], "--reportar_cada")) reportar_cada = std::atoi(sig("--reportar_cada"));
    else { std::fprintf(stderr, "Opcion desconocida: %s\n", argv[i]); return 1; }
  }

  ConjuntoMnist datos;
  std::string error;
  if (!LeerMnist(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte", &datos,
                 &error)) {
    std::fprintf(stderr, "No se pudo leer MNIST: %s\n", error.c_str());
    return 1;
  }
  const int disponibles = datos.imagenes.Shape()[0];
  const int N = (n_imagenes <= 0 || n_imagenes > disponibles) ? disponibles : n_imagenes;

  // [0, 1] -> [-1, 1] y relleno de 2 pixeles por lado con el fondo (-1). Con
  // 28×28 el latente saldria de 7×7 y la UNet2D no lo aceptaria.
  Tensor x0({N, 1, kLado, kLado});
  for (size_t i = 0; i < x0.TotalSize(); ++i) x0[i] = -1.0f;
  for (int n = 0; n < N; ++n) {
    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        x0[static_cast<size_t>(n) * kPx + static_cast<size_t>(y + 2) * kLado + (x + 2)] =
            2.0f * datos.imagenes[static_cast<size_t>(n) * 784 + static_cast<size_t>(y) * 28 + x] -
            1.0f;
      }
    }
  }

  ManualSeed(static_cast<uint32_t>(semilla));
  Codificador cod(1, canales, c_lat, grupos);
  Decodificador dec(c_lat, canales, 1, grupos);
  GaussianaDiagonal gau;

  std::vector<Tensor*> params = cod.GetParameters(), grads = cod.GetGradients();
  for (Tensor* p : dec.GetParameters()) params.push_back(p);
  for (Tensor* g : dec.GetGradients()) grads.push_back(g);
  size_t n_params = 0;
  for (const Tensor* p : params) n_params += p->TotalSize();
  AdamW opt(params, grads, lr, 0.9f, 0.999f, 1e-8f, 0.0f);

  std::printf("============================================================\n");
  std::printf("  Autoencoder convolucional (LDM-1) sobre MNIST 32x32\n");
  std::printf("  imagenes %d | lote %d | iteraciones %d | lr %.0e | peso_kl %.0e\n", N, lote,
              iteraciones, lr, peso_kl);
  std::printf("  canales %d | latente 8x8x%d (%.1fx menos posiciones que la imagen) | %zu params\n",
              canales, c_lat, static_cast<double>(kPx) / (64.0 * c_lat), n_params);
  if (N <= 64) std::printf("  MODO PUERTA: se busca memorizar\n");
  std::printf("============================================================\n");

  // Reconstruccion con la MEDIA del latente, sin sortear: es la que se usa al
  // evaluar, y la que dice que ha aprendido la red y no el azar del ruido.
  auto evaluar = [&](int cuantas) {
    const int b = std::min(cuantas, N);
    Tensor xb({b, 1, kLado, kLado});
    std::memcpy(xb.Data(), x0.Data(), static_cast<size_t>(b) * kPx * sizeof(float));
    Tensor ceros({b, c_lat, 8, 8});
    ceros.Zeros();
    GaussianaDiagonal g;
    static_cast<void>(g.Forward(cod.Forward(xb), ceros));   // con ruido 0, z = mu
    Tensor rec = dec.Forward(g.Media());
    double se = 0.0;
    for (size_t i = 0; i < rec.TotalSize(); ++i) {
      const double d = static_cast<double>(rec[i]) - xb[i];
      se += d * d;
    }
    return std::make_pair(se / static_cast<double>(rec.TotalSize()), rec);
  };

  const double mse_inicial = evaluar(N).first;
  std::printf("  antes de entrenar: MSE por pixel %.4f | PSNR %.2f dB\n\n", mse_inicial,
              Psnr(mse_inicial));

  const int b = std::min(lote, N);
  Tensor xb({b, 1, kLado, kLado}), ruido({b, c_lat, 8, 8}), drec({b, 1, kLado, kLado});
  uint32_t estado = static_cast<uint32_t>(semilla) * 2654435761u + 1u;
  auto sortear = [&]() {
    estado ^= estado << 13; estado ^= estado >> 17; estado ^= estado << 5;
    return static_cast<int>(estado % static_cast<uint32_t>(N));
  };
  const auto t0 = std::chrono::steady_clock::now();

  for (int it = 1; it <= iteraciones; ++it) {
    for (int k = 0; k < b; ++k) {
      std::memcpy(&xb[static_cast<size_t>(k) * kPx], &x0[static_cast<size_t>(sortear()) * kPx],
                  kPx * sizeof(float));
    }
    ruido.RandomNormal(0.0f, 1.0f);

    const Tensor z = gau.Forward(cod.Forward(xb), ruido);
    const Tensor rec = dec.Forward(z);

    double se = 0.0;
    for (size_t i = 0; i < rec.TotalSize(); ++i) {
      const double d = static_cast<double>(rec[i]) - xb[i];
      se += d * d;
      drec[i] = static_cast<float>(2.0 * d / b);      // d(sum/N)/d rec
    }
    const double kl = gau.KL();

    opt.ZeroGrad();
    const Tensor dz = dec.Backward(drec);
    cod.Backward(gau.Backward(dz, peso_kl));
    opt.Step();

    if (it == 1 || it % reportar_cada == 0) {
      const double mse = se / static_cast<double>(rec.TotalSize());
      const double seg =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  iter %5d/%d | MSE/pixel %.5f | PSNR %6.2f dB | KL %9.1f | %.1f s\n", it,
                  iteraciones, mse, Psnr(mse), kl, seg);
    }
  }

  auto [mse_final, rec] = evaluar(N);
  std::printf("\n  despues de entrenar (media del latente): MSE por pixel %.5f | PSNR %.2f dB\n",
              mse_final, Psnr(mse_final));

  // PNG: filas alternas de originales y reconstrucciones, 8 por fila.
  {
    const int m = std::min(N, 16), cols = 8, esc = 3, celda = kLado * esc, borde = 2 * esc;
    const int filas = 2 * ((m + cols - 1) / cols);
    image::Bitmap bmp;
    bmp.channels = 1;
    bmp.width = cols * celda + (cols + 1) * borde;
    bmp.height = filas * celda + (filas + 1) * borde;
    bmp.pixels.assign(static_cast<size_t>(bmp.width) * static_cast<size_t>(bmp.height), 60);
    auto pintar = [&](const Tensor& t, int k, int fila, int col) {
      const int ox = borde + col * (celda + borde), oy = borde + fila * (celda + borde);
      for (int y = 0; y < kLado; ++y) {
        for (int x = 0; x < kLado; ++x) {
          const float v = std::min(1.0f, std::max(-1.0f, t[static_cast<size_t>(k) * kPx + y * kLado + x]));
          const uint8_t gris = static_cast<uint8_t>(std::lround((v + 1.0f) * 127.5f));
          for (int dy = 0; dy < esc; ++dy) {
            for (int dx = 0; dx < esc; ++dx) {
              bmp.pixels[static_cast<size_t>(oy + y * esc + dy) * bmp.width + (ox + x * esc + dx)] = gris;
            }
          }
        }
      }
    };
    for (int k = 0; k < m; ++k) {
      pintar(x0, k, 2 * (k / cols), k % cols);
      pintar(rec, k, 2 * (k / cols) + 1, k % cols);
    }
    std::vector<uint8_t> bytes;
    if (image::EncodePngGris(bmp, &bytes, &error)) {
      std::ofstream(png, std::ios::binary)
          .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      std::printf("  originales (filas impares) y reconstrucciones (pares) -> %s\n", png.c_str());
    }
  }
  return 0;
}
