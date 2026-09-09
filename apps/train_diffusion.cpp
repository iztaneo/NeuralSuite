// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file train_diffusion.cpp
 * @brief Entrena la U-Net de difusion (DDPM) sobre MNIST.
 *
 * El mismo programa cubre dos usos que solo se diferencian en `--n_imagenes`:
 *
 *  - **La puerta de sobreajuste** (`--n_imagenes 16`): memorizar a proposito un
 *    punado de digitos. Si la red *no puede* memorizarlos, hay un defecto en la
 *    arquitectura, en el backward o en el entrenamiento, y ninguna cantidad de
 *    epocas lo va a arreglar. Son minutos que ahorran una tarde.
 *  - **El entrenamiento de verdad** (`--n_imagenes 0`, todo el conjunto).
 *
 * Sobre como leer la perdida: el objetivo es predecir el ruido `eps ~ N(0,1)`,
 * asi que **predecir cero da perdida 1.0**. Ese es el punto de comparacion, no
 * el cero. Y no baja igual en todos los pasos, pero al reves de lo que sugiere
 * la intuicion: como `eps = (x_t - sqrt(ab) x0) / sqrt(1 - ab)`, con `t` grande
 * `sqrt(ab)` tiende a cero y `eps` tiende a `x_t`, o sea que la red casi puede
 * copiar su entrada y la perdida baja sola. Con `t` pequeno `sqrt(1 - ab)` es
 * diminuto y hay que dividir una diferencia pequena entre un numero pequeno:
 * ahi si hace falta conocer `x0` con precision, y ahi es donde se ve si la red
 * aprendio. Lo dificil con `t` grande es predecir `x0`, no `eps`. Por eso el
 * informe parte la perdida por franjas: un unico numero promedia dos regimenes
 * distintos y esconde justo lo que la puerta tiene que mirar.
 */

#include "neuralsuite.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace neuralsuite;
using namespace neuralsuite::diffusion;
using namespace neuralsuite::data;

namespace {

/** @brief Generador propio, para no depender del RNG global de los tensores. */
struct Rng {
  uint32_t s;
  uint32_t Next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
  int Entero(int n) { return static_cast<int>(Next() % static_cast<uint32_t>(n)); }
};

/** @brief Dibuja un digito en ASCII para poder mirarlo sin salir de la consola. */
void Dibujar(const Tensor& img, size_t desplazamiento, int h, int w) {
  const char* escala = " .:-=+*#%@";
  for (int y = 0; y < h; y += 2) {
    std::string fila;
    for (int x = 0; x < w; ++x) {
      float v = img[desplazamiento + static_cast<size_t>(y) * w + x];  // en [-1, 1]
      int k = static_cast<int>((v + 1.0f) * 0.5f * 9.0f);
      if (k < 0) k = 0;
      if (k > 9) k = 9;
      fila += escala[k];
    }
    std::printf("    %s\n", fila.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "corpus/mnist";
  int n_imagenes = 16, iteraciones = 400, lote = 8, canales = 32, dim_t = 64;
  int pasos = 200, grupos = 8, semilla = 7, reportar_cada = 25;
  float lr = 2e-3f;

  for (int i = 1; i < argc; ++i) {
    auto sig = [&](const char* q) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Falta el valor de %s\n", q); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--dir")) dir = sig("--dir");
    else if (!std::strcmp(argv[i], "--n_imagenes")) n_imagenes = std::atoi(sig("--n_imagenes"));
    else if (!std::strcmp(argv[i], "--iteraciones")) iteraciones = std::atoi(sig("--iteraciones"));
    else if (!std::strcmp(argv[i], "--lote")) lote = std::atoi(sig("--lote"));
    else if (!std::strcmp(argv[i], "--canales")) canales = std::atoi(sig("--canales"));
    else if (!std::strcmp(argv[i], "--pasos")) pasos = std::atoi(sig("--pasos"));
    else if (!std::strcmp(argv[i], "--lr")) lr = static_cast<float>(std::atof(sig("--lr")));
    else if (!std::strcmp(argv[i], "--semilla")) semilla = std::atoi(sig("--semilla"));
    else if (!std::strcmp(argv[i], "--reportar_cada")) reportar_cada = std::atoi(sig("--reportar_cada"));
    else { std::fprintf(stderr, "Opcion desconocida: %s\n", argv[i]); return 1; }
  }

  ConjuntoMnist datos;
  std::string error;
  if (!LeerMnist(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte",
                 &datos, &error)) {
    std::fprintf(stderr, "No se pudo leer MNIST: %s\n", error.c_str());
    return 1;
  }
  const int H = 28, W = 28, PX = H * W;
  const int disponibles = datos.imagenes.Shape()[0];
  const int N = (n_imagenes <= 0 || n_imagenes > disponibles) ? disponibles : n_imagenes;

  // MNIST llega en [0, 1]; la difusion asume datos centrados en cero, asi que
  // se pasa a [-1, 1]. Si no, la red tendria que aprender el desplazamiento y
  // el ruido gaussiano no encajaria con la varianza que el schedule supone.
  Tensor x0({N, 1, H, W});
  for (int i = 0; i < N; ++i) {
    for (int p = 0; p < PX; ++p) {
      x0[static_cast<size_t>(i) * PX + p] =
          2.0f * datos.imagenes[static_cast<size_t>(i) * PX + p] - 1.0f;
    }
  }

  std::printf("============================================================\n");
  std::printf("  Difusion sobre MNIST\n");
  std::printf("  imagenes %d de %d | lote %d | iteraciones %d\n", N, disponibles, lote, iteraciones);
  std::printf("  canales %d | pasos de ruido %d | lr %.0e | semilla %d\n", canales, pasos, lr, semilla);
  if (N <= 64) std::printf("  MODO PUERTA: se busca memorizar, no generalizar\n");
  std::printf("============================================================\n");

  ManualSeed(static_cast<uint32_t>(semilla));
  UNet2D unet(1, canales, dim_t, grupos);
  DiffusionSchedule schedule(pasos);
  AdamW opt(unet.GetParameters(), unet.GetGradients(), lr);
  std::printf("  parametros: %zu\n\n", unet.NumParametros());

  Rng rng{static_cast<uint32_t>(semilla) * 2654435761u + 1u};
  const int lote_real = std::min(lote, N);

  Tensor xb({lote_real, 1, H, W}), eps({lote_real, 1, H, W}), xt, t({lote_real});
  Tensor dout({lote_real, 1, H, W});
  const size_t elems = xb.TotalSize();
  const auto t0 = std::chrono::steady_clock::now();

  // Franjas de t, para no promediar dos regimenes distintos en un solo numero.
  double sum_baja = 0, sum_alta = 0; int n_baja = 0, n_alta = 0;

  for (int it = 1; it <= iteraciones; ++it) {
    for (int b = 0; b < lote_real; ++b) {
      const int idx = rng.Entero(N);
      std::memcpy(&xb[static_cast<size_t>(b) * PX], &x0[static_cast<size_t>(idx) * PX],
                  static_cast<size_t>(PX) * sizeof(float));
      t[b] = static_cast<float>(rng.Entero(pasos));
    }
    eps.RandomNormal(0.0f, 1.0f);
    schedule.QSample(xb, eps, t, &xt);

    const Tensor pred = unet.Forward(xt, t);

    // MSE contra el ruido, con su derivada: 2 (pred - eps) / n.
    double perdida = 0.0;
    for (size_t i = 0; i < elems; ++i) {
      const double d = static_cast<double>(pred[i]) - eps[i];
      perdida += d * d;
      dout[i] = static_cast<float>(2.0 * d / static_cast<double>(elems));
    }
    perdida /= static_cast<double>(elems);

    for (int b = 0; b < lote_real; ++b) {
      if (t[b] < pasos / 2) { sum_baja += perdida; ++n_baja; }
      else { sum_alta += perdida; ++n_alta; }
    }

    opt.ZeroGrad();
    unet.Backward(dout);
    opt.Step();

    if (it % reportar_cada == 0 || it == 1) {
      const double seg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  iter %5d/%d | perdida %.4f | t bajo %.4f | t alto %.4f | %.1f s\n",
                  it, iteraciones, perdida,
                  n_baja ? sum_baja / n_baja : 0.0, n_alta ? sum_alta / n_alta : 0.0, seg);
      sum_baja = sum_alta = 0; n_baja = n_alta = 0;
    }
  }

  // Evaluacion final por franjas, con ruido nuevo y t barrido de forma
  // determinista: la perdida de entrenamiento se mide sobre lo que toco el
  // optimizador y siempre parece mejor de lo que es.
  std::printf("\n  --- perdida final por franja de t (predecir cero da 1.0) ---\n");
  const int franjas = 4, por_franja = pasos / franjas;
  for (int f = 0; f < franjas; ++f) {
    double acc = 0.0; int veces = 0;
    for (int rep = 0; rep < 8; ++rep) {
      for (int b = 0; b < lote_real; ++b) {
        std::memcpy(&xb[static_cast<size_t>(b) * PX],
                    &x0[static_cast<size_t>(rng.Entero(N)) * PX],
                    static_cast<size_t>(PX) * sizeof(float));
        t[b] = static_cast<float>(f * por_franja + rng.Entero(por_franja));
      }
      eps.RandomNormal(0.0f, 1.0f);
      schedule.QSample(xb, eps, t, &xt);
      const Tensor pred = unet.Forward(xt, t);
      double l = 0.0;
      for (size_t i = 0; i < elems; ++i) {
        const double d = static_cast<double>(pred[i]) - eps[i];
        l += d * d;
      }
      acc += l / static_cast<double>(elems);
      ++veces;
    }
    std::printf("  t en [%3d, %3d) : %.4f\n", f * por_franja, (f + 1) * por_franja, acc / veces);
  }

  // La prueba visual, con su control. Mirar la reconstruccion a `t` bajo no
  // demuestra nada: ahi `x_t` ya es casi la imagen limpia y **una red sin
  // entrenar tambien devuelve el digito**, porque `PredecirX0` esta deshaciendo
  // un ruido que apenas tapaba nada. Se comprobo: a t=20 la red recien
  // inicializada dibujaba el mismo cinco. Asi que el dibujo se hace donde `x_t`
  // ya es irreconocible, y al lado se pone lo que da una red sin entrenar con
  // la misma entrada. Si las dos columnas se parecen, no se aprendio nada.
  {
    const int t_vista = 7 * pasos / 10;
    std::printf("\n  --- x_0 despejado desde t=%d ---\n", t_vista);
    Tensor uno({1, 1, H, W}), r({1, 1, H, W}), tu({1}), xtu, x0p, x0c;
    std::memcpy(&uno[0], &x0[0], static_cast<size_t>(PX) * sizeof(float));
    r.RandomNormal(0.0f, 1.0f);
    tu[0] = static_cast<float>(t_vista);
    schedule.QSample(uno, r, tu, &xtu);

    const Tensor pred = unet.Forward(xtu, tu);
    schedule.PredecirX0(xtu, pred, tu, &x0p);

    UNet2D sin_entrenar(1, canales, dim_t, grupos);
    const Tensor pred_c = sin_entrenar.Forward(xtu, tu);
    schedule.PredecirX0(xtu, pred_c, tu, &x0c);

    std::printf("\n  original:\n");                 Dibujar(uno, 0, H, W);
    std::printf("\n  x_t, lo que ve la red:\n");    Dibujar(xtu, 0, H, W);
    std::printf("\n  x_0 segun la red entrenada:\n"); Dibujar(x0p, 0, H, W);
    std::printf("\n  CONTROL, x_0 segun una red sin entrenar:\n"); Dibujar(x0c, 0, H, W);
  }
  std::printf("\n");
  return 0;
}
