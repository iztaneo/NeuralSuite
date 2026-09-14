// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file sample_diffusion.cpp
 * @brief Genera digitos con una U-Net de difusion ya entrenada y los guarda en PNG.
 *
 * Es el examen del escalon 5, separado del entrenamiento a proposito: se puede
 * repetir sobre cualquier checkpoint —el final o uno archivado— sin volver a
 * entrenar nada, y con tantas muestras como haga falta para juzgar.
 *
 * Dos cosas que el log del entrenamiento no podia dar:
 *
 *  - **Resolucion completa.** El ASCII salta una fila de cada dos y esconde
 *    justo el detalle que decide si un digito es legible.
 *  - **Diversidad.** Cuatro muestras no dicen si salen las diez clases o si el
 *    modelo se ha quedado con dos o tres formas. Aqui se asigna a cada muestra
 *    la etiqueta de su vecina mas cercana en MNIST. Es un clasificador tosco
 *    —la distancia en pixeles no es la forma—, pero basta para ver un colapso:
 *    si 64 muestras caen en tres etiquetas, algo va mal.
 *
 * La misma vecina da la otra medida del examen: la distancia a ella comparada
 * con la distancia media al conjunto. Si es muy pequena, el modelo copia en vez
 * de generar.
 */

#include "neuralsuite.h"
#include "image/png.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <map>
#include <string>
#include <vector>

using namespace neuralsuite;
using namespace neuralsuite::diffusion;
using namespace neuralsuite::data;

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  std::string archivo = "release/unet_mnist.nsf.ema";
  std::string salida = "release/muestras_mnist.png";
  std::string dir = "corpus/mnist";
  std::string muestreador = "ddim";
  int n = 64, canales = 32, dim_t = 64, grupos = 8, pasos = 1000, pasos_ddim = 100;
  int semilla = 2026, escala = 4;
  float beta_fin = 0.0f, eta = 0.0f;
  bool recortar = true;

  std::set<std::string> dados;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--", 2) == 0) dados.insert(argv[i] + 2);
    auto sig = [&](const char* q) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Falta el valor de %s\n", q); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--archivo")) archivo = sig("--archivo");
    else if (!std::strcmp(argv[i], "--salida")) salida = sig("--salida");
    else if (!std::strcmp(argv[i], "--dir")) dir = sig("--dir");
    else if (!std::strcmp(argv[i], "--muestreador")) muestreador = sig("--muestreador");
    else if (!std::strcmp(argv[i], "--n")) n = std::atoi(sig("--n"));
    else if (!std::strcmp(argv[i], "--canales")) canales = std::atoi(sig("--canales"));
    else if (!std::strcmp(argv[i], "--pasos")) pasos = std::atoi(sig("--pasos"));
    else if (!std::strcmp(argv[i], "--pasos_ddim")) pasos_ddim = std::atoi(sig("--pasos_ddim"));
    else if (!std::strcmp(argv[i], "--beta_fin")) beta_fin = static_cast<float>(std::atof(sig("--beta_fin")));
    else if (!std::strcmp(argv[i], "--semilla")) semilla = std::atoi(sig("--semilla"));
    else if (!std::strcmp(argv[i], "--escala")) escala = std::atoi(sig("--escala"));
    else if (!std::strcmp(argv[i], "--eta")) eta = static_cast<float>(std::atof(sig("--eta")));
    else if (!std::strcmp(argv[i], "--sin_recorte")) recortar = false;
    else { std::fprintf(stderr, "Opcion desconocida: %s\n", argv[i]); return 1; }
  }
  if (muestreador != "ddim" && muestreador != "ddpm") {
    std::fprintf(stderr, "--muestreador debe ser ddim o ddpm\n");
    return 1;
  }
  if (n <= 0 || escala <= 0) {
    std::fprintf(stderr, "--n y --escala deben ser positivos\n");
    return 1;
  }

  // El calendario sale del sello del checkpoint. Muestrear con otro calendario
  // no da ningun error —los pesos cargan igual— y las imagenes salen peores sin
  // decir por que, asi que no se deja a la memoria de quien lo ejecuta. Si se
  // pide explicitamente algo distinto de lo sellado, se aborta.
  float beta_ini = 1e-4f;
  {
    std::map<std::string, std::string> sellado;
    const auto r = nsf::ReadMetadata(archivo, &sellado);
    if (!r) {
      std::fprintf(stderr, "No se pudo leer %s: %s\n", archivo.c_str(), r.error.c_str());
      return 1;
    }
    if (sellado.count("pasos") && sellado.count("beta_fin")) {
      auto adoptar_int = [&](const char* k, int* v) {
        const int sel = std::stoi(sellado[k]);
        if (dados.count(k) && *v != sel) {
          std::fprintf(stderr, "El checkpoint se entreno con %s=%d y se pidio %d\n", k, sel, *v);
          std::exit(1);
        }
        *v = sel;
      };
      adoptar_int("pasos", &pasos);
      adoptar_int("canales", &canales);
      const float sel_beta = std::stof(sellado["beta_fin"]);
      if (dados.count("beta_fin") && std::abs(beta_fin - sel_beta) > 1e-7f) {
        std::fprintf(stderr, "El checkpoint se entreno con beta_fin=%s y se pidio %g\n",
                     sellado["beta_fin"].c_str(), beta_fin);
        return 1;
      }
      beta_fin = sel_beta;
      if (sellado.count("beta_ini")) beta_ini = std::stof(sellado["beta_ini"]);
      if (sellado.count("normalizacion") && sellado["normalizacion"] != "[-1,1]") {
        std::fprintf(stderr, "El checkpoint normaliza a %s y este programa asume [-1,1]\n",
                     sellado["normalizacion"].c_str());
        return 1;
      }
      std::printf("calendario sellado en el checkpoint\n");
    } else {
      // Checkpoints anteriores al sello: se reconstruye con la regla del
      // entrenador, y se avisa, porque aqui ya no hay garantia.
      if (beta_fin <= 0.0f) beta_fin = std::min(0.5f, 0.02f * 1000.0f / static_cast<float>(pasos));
      std::printf("AVISO: el checkpoint no sella su calendario; se asume %d pasos y beta final "
                  "%.4f. Si se entreno con otro, las muestras saldran peor sin error.\n",
                  pasos, beta_fin);
    }
  }
  DiffusionSchedule calendario(pasos, beta_ini, beta_fin);

  UNet2D unet(1, canales, dim_t, grupos);
  if (!unet.CargarPesos(archivo)) {
    std::fprintf(stderr, "No se pudieron cargar los pesos de %s\n", archivo.c_str());
    return 1;
  }
  std::printf("pesos %s | calendario %d pasos, beta final %.4f\n", archivo.c_str(), pasos,
              beta_fin);

  const int H = 28, W = 28, PX = H * W;
  ManualSeed(static_cast<uint32_t>(semilla));
  Tensor ruido({n, 1, H, W});
  ruido.RandomNormal(0.0f, 1.0f);
  Predictor red = [&](const Tensor& x, const Tensor& t) { return unet.Forward(x, t); };

  const auto t0 = std::chrono::steady_clock::now();
  Tensor muestras;
  try {
  if (muestreador == "ddim") {
    // Con eta > 0 DDIM vuelve a meter ruido en cada paso; con eta = 1 y todos
    // los pasos equivale a DDPM. Tenerlo aqui permite separar dos causas de
    // mala calidad que con eta fijo se confunden: pocos pasos, o determinismo.
    DDIMSampler ddim(calendario, pasos_ddim, eta);
    ddim.RecortarX0(recortar);
    muestras = ddim.Muestrear(red, ruido, RuidoGaussiano(static_cast<uint32_t>(semilla)));
  } else {
    DDPMSampler ddpm(calendario);
    ddpm.RecortarX0(recortar);
    muestras = ddpm.Muestrear(red, ruido, RuidoGaussiano(static_cast<uint32_t>(semilla)));
  }
  } catch (const std::invalid_argument& e) {
    // Argumentos que el muestreador rechaza, como un DDIM de un solo paso: se
    // explica y se sale, en vez de abortar con una excepcion sin capturar.
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  const double seg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (muestreador == "ddim") {
    std::printf("%d muestras con ddim (%d pasos, eta %.2f) en %.1f s\n", n, pasos_ddim, eta, seg);
  } else {
    std::printf("%d muestras con ddpm (%d pasos) en %.1f s\n", n, pasos, seg);
  }

  // --- Rejilla en PNG. Casilla de 28*escala pixeles con un borde de 2*escala.
  const int lado = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
  const int celda = W * escala, borde = 2 * escala;
  image::Bitmap bmp;
  bmp.channels = 1;
  bmp.width = lado * celda + (lado + 1) * borde;
  bmp.height = bmp.width;
  bmp.pixels.assign(static_cast<size_t>(bmp.width) * static_cast<size_t>(bmp.height), 60);
  for (int k = 0; k < n; ++k) {
    const int ox = borde + (k % lado) * (celda + borde);
    const int oy = borde + (k / lado) * (celda + borde);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float v = muestras[static_cast<size_t>(k) * PX + static_cast<size_t>(y) * W + x];
        v = std::min(1.0f, std::max(-1.0f, v));
        const uint8_t g = static_cast<uint8_t>(std::lround((v + 1.0f) * 127.5f));
        for (int dy = 0; dy < escala; ++dy) {
          for (int dx = 0; dx < escala; ++dx) {
            bmp.pixels[static_cast<size_t>(oy + y * escala + dy) * static_cast<size_t>(bmp.width) +
                       static_cast<size_t>(ox + x * escala + dx)] = g;
          }
        }
      }
    }
  }
  std::vector<uint8_t> png;
  std::string error;
  if (!image::EncodePngGris(bmp, &png, &error)) {
    std::fprintf(stderr, "No se pudo codificar el PNG: %s\n", error.c_str());
    return 1;
  }
  std::ofstream(salida, std::ios::binary)
      .write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
  std::printf("rejilla %dx%d -> %s\n", bmp.width, bmp.height, salida.c_str());

  // --- Vecina mas cercana en MNIST: etiqueta aproximada y distancia.
  ConjuntoMnist datos;
  if (!LeerMnist(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte", &datos,
                 &error)) {
    std::fprintf(stderr, "Sin MNIST no hay vecinas: %s\n", error.c_str());
    return 0;
  }
  const int N = datos.imagenes.Shape()[0];
  std::vector<int> por_clase(10, 0);
  double suma_cercana = 0.0, suma_media = 0.0;
  for (int k = 0; k < n; ++k) {
    std::vector<double> dist(static_cast<size_t>(N));
    parallel::ParallelFor(N, 512, [&](int ini, int fin) {
      for (int i = ini; i < fin; ++i) {
        double d = 0.0;
        for (int p = 0; p < PX; ++p) {
          const double e = muestras[static_cast<size_t>(k) * PX + p] -
                           (2.0 * datos.imagenes[static_cast<size_t>(i) * PX + p] - 1.0);
          d += e * e;
        }
        dist[static_cast<size_t>(i)] = std::sqrt(d / PX);
      }
    });
    int mejor = 0;
    double media = 0.0;
    for (int i = 0; i < N; ++i) {
      media += dist[static_cast<size_t>(i)];
      if (dist[static_cast<size_t>(i)] < dist[static_cast<size_t>(mejor)]) mejor = i;
    }
    media /= N;
    ++por_clase[static_cast<size_t>(datos.etiquetas[mejor])];
    suma_cercana += dist[static_cast<size_t>(mejor)];
    suma_media += media;
  }
  std::printf("\netiqueta de la vecina mas cercana (clasificador tosco):\n ");
  for (int c = 0; c < 10; ++c) std::printf("  %d:%-3d", c, por_clase[static_cast<size_t>(c)]);
  int clases = 0;
  for (int c : por_clase) clases += c > 0 ? 1 : 0;
  std::printf("\n  clases distintas: %d de 10\n", clases);
  std::printf("distancia a la vecina mas cercana %.4f | media al conjunto %.4f | cociente %.2f\n",
              suma_cercana / n, suma_media / n, suma_cercana / suma_media);
  std::printf("(en la puerta de sobreajuste, que copiaba, el cociente rondaba 0.15-0.35)\n");
  return 0;
}
