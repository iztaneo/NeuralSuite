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
 * `peso_kl` del paper (1e-6) no significaria lo mismo aqui. Error cuadratico y
 * no el L1 + perdida perceptual del paper, porque la perceptual necesita una
 * VGG preentrenada, y esa decision esta aplazada.
 *
 * **Como se juzga.** La puerta de sobreajuste enseno que un PSNR a secas no
 * dice nada: un codificador sin entrenar ya daba 21 dB. Por eso cada
 * evaluacion se compara con un compresor tonto que guarda **los mismos numeros
 * que el latente**: reducir la imagen por promedio a `64·C` pixeles y volver a
 * ampliarla por interpolacion bilineal (8×8 con C=1, 8×16 con C=2, 16×16 con
 * C=4). Si el autoencoder no gana claramente a eso, no ha aprendido nada que
 * valga la pena.
 *
 * La infraestructura de un run largo —sello, checkpoint transaccional, estado
 * de Adam, tasa con calentamiento y coseno, semilla por iteracion— es la de
 * `entrenamiento/checkpoint.h`, compartida con `train_diffusion`.
 */

#include "neuralsuite.h"
#include "image/bitmap.h"
#include "image/png.h"

#include <array>
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

/** @brief Tamano del compresor tonto con los mismos `64·C` numeros que el latente. */
std::pair<int, int> TamanoReferencia(int c_lat) {
  int alto = 8, ancho = 8 * c_lat;
  while (ancho > alto * 2 && alto < kLado) { alto *= 2; ancho /= 2; }
  return {alto, ancho};
}

/**
 * @brief Error cuadratico medio por pixel de reducir por promedio y ampliar.
 */
double MseReferencia(const Tensor& imagenes, int cuantas, int alto, int ancho) {
  const int fy = kLado / alto, fx = kLado / ancho;
  double se = 0.0;
  std::vector<float> chica(static_cast<size_t>(alto) * ancho), grande;
  for (int n = 0; n < cuantas; ++n) {
    const size_t base = static_cast<size_t>(n) * kPx;
    for (int y = 0; y < alto; ++y) {
      for (int x = 0; x < ancho; ++x) {
        double acc = 0.0;
        for (int dy = 0; dy < fy; ++dy) {
          for (int dx = 0; dx < fx; ++dx) acc += imagenes[base + (y * fy + dy) * kLado + (x * fx + dx)];
        }
        chica[static_cast<size_t>(y) * ancho + x] = static_cast<float>(acc / (fy * fx));
      }
    }
    image::Resize(chica, ancho, alto, &grande, kLado, kLado);
    for (int p = 0; p < kPx; ++p) {
      const double d = static_cast<double>(grande[static_cast<size_t>(p)]) - imagenes[base + p];
      se += d * d;
    }
  }
  return se / (static_cast<double>(cuantas) * kPx);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  std::string dir = "corpus/mnist";
  std::string png = "release/autoencoder_reconstruccion.png";
  std::string archivo = "release/autoencoder_mnist.nsf";
  std::string particion = "barajada";
  int n_imagenes = 16, n_validacion = 0, iteraciones = 600, lote = 16, canales = 32, c_lat = 4;
  int grupos = 8, semilla = 7, reportar_cada = 50, calentamiento = 0;
  int evaluar_cada = 0, guardar_cada = 0, archivar_cada = 0, parar_en = 0, n_eval = 1000;
  float lr = 1e-3f, lr_min = 0.0f, peso_kl = 1e-6f;
  bool reanudar = false, medir_escala = false;
  std::string escala_txt, media_txt;

  entrenamiento::RegistroSellado sellables;
  sellables.MarcarDados(argc, argv);
  for (int i = 1; i < argc; ++i) {
    auto sig = [&](const char* q) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "Falta el valor de %s\n", q); std::exit(1); }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--dir")) dir = sig("--dir");
    else if (!std::strcmp(argv[i], "--png")) png = sig("--png");
    else if (!std::strcmp(argv[i], "--archivo")) archivo = sig("--archivo");
    else if (!std::strcmp(argv[i], "--particion")) particion = sig("--particion");
    else if (!std::strcmp(argv[i], "--n_imagenes")) n_imagenes = std::atoi(sig("--n_imagenes"));
    else if (!std::strcmp(argv[i], "--n_validacion")) n_validacion = std::atoi(sig("--n_validacion"));
    else if (!std::strcmp(argv[i], "--iteraciones")) iteraciones = std::atoi(sig("--iteraciones"));
    else if (!std::strcmp(argv[i], "--lote")) lote = std::atoi(sig("--lote"));
    else if (!std::strcmp(argv[i], "--canales")) canales = std::atoi(sig("--canales"));
    else if (!std::strcmp(argv[i], "--c_lat")) c_lat = std::atoi(sig("--c_lat"));
    else if (!std::strcmp(argv[i], "--lr")) lr = static_cast<float>(std::atof(sig("--lr")));
    else if (!std::strcmp(argv[i], "--lr_min")) lr_min = static_cast<float>(std::atof(sig("--lr_min")));
    else if (!std::strcmp(argv[i], "--calentamiento")) calentamiento = std::atoi(sig("--calentamiento"));
    else if (!std::strcmp(argv[i], "--peso_kl")) peso_kl = static_cast<float>(std::atof(sig("--peso_kl")));
    else if (!std::strcmp(argv[i], "--semilla")) semilla = std::atoi(sig("--semilla"));
    else if (!std::strcmp(argv[i], "--reportar_cada")) reportar_cada = std::atoi(sig("--reportar_cada"));
    else if (!std::strcmp(argv[i], "--evaluar_cada")) evaluar_cada = std::atoi(sig("--evaluar_cada"));
    else if (!std::strcmp(argv[i], "--guardar_cada")) guardar_cada = std::atoi(sig("--guardar_cada"));
    else if (!std::strcmp(argv[i], "--archivar_cada")) archivar_cada = std::atoi(sig("--archivar_cada"));
    else if (!std::strcmp(argv[i], "--parar_en")) parar_en = std::atoi(sig("--parar_en"));
    else if (!std::strcmp(argv[i], "--n_eval")) n_eval = std::atoi(sig("--n_eval"));
    else if (!std::strcmp(argv[i], "--reanudar")) reanudar = true;
    else if (!std::strcmp(argv[i], "--medir_escala")) { medir_escala = true; reanudar = true; }
    else { std::fprintf(stderr, "Opcion desconocida: %s\n", argv[i]); return 1; }
  }

  // Todo lo que decide la trayectoria va sellado; al reanudar manda el sello.
  sellables.Entero("n_imagenes", &n_imagenes);
  sellables.Entero("n_validacion", &n_validacion);
  sellables.Texto("particion", &particion);
  sellables.Entero("semilla", &semilla);
  sellables.Entero("canales", &canales);
  sellables.Entero("c_lat", &c_lat);
  sellables.Entero("lote", &lote);
  sellables.Entero("iteraciones", &iteraciones);
  sellables.Entero("calentamiento", &calentamiento);
  sellables.Real("lr", &lr);
  sellables.Real("lr_min", &lr_min);
  sellables.Real("peso_kl", &peso_kl);

  const std::array<std::string, 3> rutas = {archivo, archivo + ".dec", archivo + ".opt"};
  if (reanudar) {
    entrenamiento::Metadatos sellado;
    const auto r = nsf::ReadMetadata(rutas[2], &sellado);
    if (!r) {
      std::fprintf(stderr, "No se pudo leer el sello de %s: %s\n", rutas[2].c_str(), r.error.c_str());
      return 1;
    }
    std::string adoptados, motivo;
    if (!sellables.Adoptar(sellado, &adoptados, &motivo)) {
      std::fprintf(stderr, "%s\n", motivo.c_str());
      return 1;
    }
    if (!adoptados.empty()) std::printf("  del checkpoint:%s\n", adoptados.c_str());
  }
  if (particion != "barajada" && particion != "contigua") {
    std::fprintf(stderr, "--particion debe ser barajada o contigua\n");
    return 1;
  }

  ConjuntoMnist datos;
  std::string error;
  if (!LeerMnist(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte", &datos,
                 &error)) {
    std::fprintf(stderr, "No se pudo leer MNIST: %s\n", error.c_str());
    return 1;
  }
  const int disponibles = datos.imagenes.Shape()[0];
  const int total = (n_imagenes <= 0 || n_imagenes > disponibles) ? disponibles : n_imagenes;
  if (n_validacion < 0 || n_validacion >= total) {
    std::fprintf(stderr, "n_validacion debe estar entre 0 y %d\n", total - 1);
    return 1;
  }
  const int N = total - n_validacion;

  // [0, 1] -> [-1, 1] y relleno de 2 pixeles por lado con el fondo (-1). Con
  // 28×28 el latente saldria de 7×7 y la UNet2D no lo aceptaria.
  auto rellenar = [&](int cuantas) {
    Tensor t({cuantas, 1, kLado, kLado});
    for (size_t i = 0; i < t.TotalSize(); ++i) t[i] = -1.0f;
    for (int n = 0; n < cuantas; ++n) {
      for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
          t[static_cast<size_t>(n) * kPx + static_cast<size_t>(y + 2) * kLado + (x + 2)] =
              2.0f * datos.imagenes[static_cast<size_t>(n) * 784 + static_cast<size_t>(y) * 28 + x] - 1.0f;
        }
      }
    }
    return t;
  };
  Tensor x0, x_val;
  {
    Tensor todas = rellenar(total);
    if (particion == "contigua" || n_validacion == 0) {
      x0 = Tensor({N, 1, kLado, kLado});
      std::memcpy(x0.Data(), todas.Data(), static_cast<size_t>(N) * kPx * sizeof(float));
      x_val = Tensor({std::max(n_validacion, 1), 1, kLado, kLado});
      if (n_validacion > 0) {
        std::memcpy(x_val.Data(), todas.Data() + static_cast<size_t>(N) * kPx,
                    static_cast<size_t>(n_validacion) * kPx * sizeof(float));
      }
    } else {
      // Particion con DataLoader::Partir sobre indices barajados; su semilla
      // se deriva de la del run, asi que misma semilla, misma particion.
      Tensor etiquetas({total});
      for (int i = 0; i < total; ++i) etiquetas[i] = datos.etiquetas[i];
      const DataLoader completo(std::move(todas), std::move(etiquetas), total, true,
                                static_cast<uint32_t>(semilla) * 7919u + 13u, true);
      auto partes = completo.Partir((static_cast<float>(n_validacion) + 0.5f) / static_cast<float>(total));
      if (partes.first.Tamano() != N || partes.second.Tamano() != n_validacion) {
        std::fprintf(stderr, "La particion no dio %d + %d\n", N, n_validacion);
        return 1;
      }
      Tensor desechable;
      partes.first.Lote(0, &x0, &desechable);
      partes.second.Lote(0, &x_val, &desechable);
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

  const auto [ref_alto, ref_ancho] = TamanoReferencia(c_lat);
  std::printf("============================================================\n");
  std::printf("  Autoencoder convolucional (LDM-1) sobre MNIST 32x32\n");
  std::printf("  imagenes %d (%d validacion, particion %s) | lote %d | iteraciones %d\n", N,
              n_validacion, particion.c_str(), lote, iteraciones);
  std::printf("  lr %.0e -> %.0e, calentamiento %d | peso_kl %.0e | semilla %d\n", lr, lr_min,
              calentamiento, peso_kl, semilla);
  std::printf("  canales %d | latente 8x8x%d = %d numeros | %zu params\n", canales, c_lat,
              64 * c_lat, n_params);
  std::printf("  referencia: reducir a %dx%d y ampliar (los mismos %d numeros)\n", ref_alto,
              ref_ancho, ref_alto * ref_ancho);
  std::printf("  archivo %s%s\n", archivo.c_str(), reanudar ? " (reanudando)" : "");
  if (N <= 64) std::printf("  MODO PUERTA: se busca memorizar\n");
  std::printf("============================================================\n");

  // --- Checkpoint: codificador, decodificador y Adam, sellados y transaccionales.
  auto guardar = [&](int it, const std::string& base) {
    entrenamiento::Metadatos sello = sellables.Valores();
    sello["normalizacion"] = "[-1,1]";
    sello["relleno"] = "32";
    if (!escala_txt.empty()) {
      sello["escala_latente"] = escala_txt;
      sello["media_latente"] = media_txt;
    }
    const std::vector<entrenamiento::Parte> partes = {
        {base, [&](const std::string& r, const entrenamiento::Metadatos& m) { return cod.GuardarPesos(r, m); }},
        {base + ".dec", [&](const std::string& r, const entrenamiento::Metadatos& m) { return dec.GuardarPesos(r, m); }},
        {base + ".opt", [&](const std::string& r, const entrenamiento::Metadatos& m) {
           return entrenamiento::GuardarEstadoAdam(r, opt, m).ok;
         }},
    };
    std::string motivo;
    if (!entrenamiento::GuardarCheckpoint(partes, it, sello, &motivo)) {
      std::fprintf(stderr, "  checkpoint descartado (%s); el anterior sigue en pie\n", motivo.c_str());
      return false;
    }
    return true;
  };

  int it_inicial = 1;
  if (reanudar) {
    entrenamiento::Metadatos m_cod, m_dec, m_opt;
    if (!cod.CargarPesos(rutas[0], &m_cod) || !dec.CargarPesos(rutas[1], &m_dec)) return 1;
    const auto r = entrenamiento::CargarEstadoAdam(rutas[2], opt, &m_opt);
    if (!r) {
      std::fprintf(stderr, "No se pudo leer el estado del optimizador: %s\n", r.error.c_str());
      return 1;
    }
    std::string motivo;
    if (!entrenamiento::ComprobarMismoCheckpoint({m_cod, m_dec, m_opt},
                                                 {"codificador", "decodificador", "optimizador"}, &motivo)) {
      std::fprintf(stderr, "%s\n", motivo.c_str());
      return 1;
    }
    it_inicial = std::stoi(m_opt["iteracion"]) + 1;
    std::printf("  reanudado en la iteracion %d (Adam %d pasos)\n", it_inicial, opt.PasosDados());
    if (m_cod.count("escala_latente")) {
      std::printf("  escala del latente ya sellada: %s\n", m_cod["escala_latente"].c_str());
    }
  }

  // --- Medir la escala del latente y sellarla, sin entrenar.
  //
  // El paper reescala el latente por 1/sigma antes de difundir sobre el. La
  // razon es que la difusion supone datos de varianza cercana a uno —su
  // calendario de ruido esta calibrado para eso— y el latente de un autoencoder
  // con KL debil no tiene por que cumplirlo: aqui la KL apenas regulariza y el
  // latente se expande durante el entrenamiento. Sin reescalar, el modelo de
  // difusion veria entradas de otra escala y el calendario dejaria de
  // corresponder, sin que nada diera error.
  //
  // Se mide con la MEDIA del latente sobre validacion, que es lo que la Fase 19
  // difundira, y se guarda en el sello de los tres archivos.
  if (medir_escala) {
    const int m = std::min(n_eval, n_validacion > 0 ? n_validacion : N);
    const Tensor& conjunto = n_validacion > 0 ? x_val : x0;
    double suma = 0.0, suma2 = 0.0;
    size_t cuenta = 0;
    for (int ini = 0; ini < m; ini += 64) {
      const int b = std::min(64, m - ini);
      Tensor xb({b, 1, kLado, kLado}), ceros({b, c_lat, 8, 8});
      std::memcpy(xb.Data(), conjunto.Data() + static_cast<size_t>(ini) * kPx,
                  static_cast<size_t>(b) * kPx * sizeof(float));
      ceros.Zeros();
      GaussianaDiagonal g;
      static_cast<void>(g.Forward(cod.Forward(xb), ceros));
      const Tensor& mu = g.Media();
      for (size_t i = 0; i < mu.TotalSize(); ++i) {
        suma += mu[i];
        suma2 += static_cast<double>(mu[i]) * mu[i];
        ++cuenta;
      }
    }
    const double media = suma / static_cast<double>(cuenta);
    const double sigma = std::sqrt(suma2 / static_cast<double>(cuenta) - media * media);
    escala_txt = entrenamiento::TextoReal(1.0 / sigma);
    // Se sella tambien la MEDIA, que el paper no necesita. Alli los latentes ya
    // salen centrados y basta con dividir por sigma; aqui la media es -0.69 con
    // sigma 0.61, asi que reescalar sin centrar dejaria el latente desplazado
    // mas de una desviacion. La difusion supone datos centrados —su calendario
    // esta calibrado para eso— y un desplazamiento asi no da error: da imagenes
    // peores. La Fase 19 usara (z - media) * escala.
    media_txt = entrenamiento::TextoReal(media);
    std::printf("  latente sobre %d imagenes: media %.4f, sigma %.4f\n", m, media, sigma);
    std::printf("  sellado para la Fase 19: (z - %s) * %s\n", media_txt.c_str(),
                escala_txt.c_str());

    // Y la comprobacion que exige el criterio de salida de la fase: que la
    // UNet2D acepte este latente tal cual, sin adaptadores.
    {
      diffusion::UNet2D prueba(c_lat, 32, 64, 8);
      Tensor z({2, c_lat, 8, 8}), t({2});
      z.RandomNormal(0.0f, 1.0f);
      t[0] = 5.0f;
      t[1] = 700.0f;
      const Tensor salida = prueba.Forward(z, t);
      const bool ok = salida.Shape() == z.Shape();
      std::printf("  la UNet2D acepta el latente 8x8x%d y devuelve [%d,%d,%d,%d]  %s\n", c_lat,
                  salida.Shape()[0], salida.Shape()[1], salida.Shape()[2], salida.Shape()[3],
                  ok ? "ok" : "MAL");
      if (!ok) return 1;
    }
    if (!guardar(it_inicial - 1, archivo)) return 1;
    std::printf("  escala sellada en %s (+ .dec, + .opt)\n", archivo.c_str());
    return 0;
  }

  // Reconstruccion con la MEDIA del latente, sin sortear: es la que se usa al
  // evaluar, y la que dice lo que ha aprendido la red y no el azar del ruido.
  // Se procesa por trozos para no reservar el conjunto entero de golpe.
  auto reconstruir = [&](const Tensor& conjunto, int cuantas, Tensor* salida) {
    double se = 0.0;
    if (salida) *salida = Tensor({cuantas, 1, kLado, kLado});
    for (int ini = 0; ini < cuantas; ini += 64) {
      const int b = std::min(64, cuantas - ini);
      Tensor xb({b, 1, kLado, kLado}), ceros({b, c_lat, 8, 8});
      std::memcpy(xb.Data(), conjunto.Data() + static_cast<size_t>(ini) * kPx,
                  static_cast<size_t>(b) * kPx * sizeof(float));
      ceros.Zeros();
      GaussianaDiagonal g;
      static_cast<void>(g.Forward(cod.Forward(xb), ceros));
      const Tensor rec = dec.Forward(g.Media());
      for (size_t i = 0; i < rec.TotalSize(); ++i) {
        const double d = static_cast<double>(rec[i]) - xb[i];
        se += d * d;
      }
      if (salida) std::memcpy(salida->Data() + static_cast<size_t>(ini) * kPx, rec.Data(), rec.TotalSize() * sizeof(float));
    }
    return se / (static_cast<double>(cuantas) * kPx);
  };

  auto guardar_png = [&](const Tensor& originales, const Tensor& recs, int m) {
    const int cols = 8, esc = 3, celda = kLado * esc, borde = 2 * esc;
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
      pintar(originales, k, 2 * (k / cols), k % cols);
      pintar(recs, k, 2 * (k / cols) + 1, k % cols);
    }
    std::vector<uint8_t> bytes;
    std::string e;
    if (image::EncodePngGris(bmp, &bytes, &e)) {
      std::ofstream(png, std::ios::binary)
          .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
  };

  // Lo que se evalua: la validacion si la hay, y si no el propio entrenamiento.
  const Tensor& conjunto_eval = n_validacion > 0 ? x_val : x0;
  const int m_eval = std::min(n_eval, n_validacion > 0 ? n_validacion : N);
  const double mse_ref = MseReferencia(conjunto_eval, m_eval, ref_alto, ref_ancho);
  auto evaluar = [&](int it) {
    Tensor recs;
    const double mse_va = reconstruir(conjunto_eval, m_eval, &recs);
    const double mse_tr = reconstruir(x0, std::min(m_eval, N), nullptr);
    std::printf("  eval %6d | PSNR entren. %6.2f | %s %6.2f | referencia %6.2f | ventaja %+6.2f dB\n",
                it, Psnr(mse_tr), n_validacion > 0 ? "valid." : "entren.", Psnr(mse_va),
                Psnr(mse_ref), Psnr(mse_va) - Psnr(mse_ref));
    guardar_png(conjunto_eval, recs, std::min(m_eval, 16));
  };
  if (!reanudar) evaluar(0);

  const int b = std::min(lote, N);
  Tensor xb({b, 1, kLado, kLado}), ruido({b, c_lat, 8, 8}), drec({b, 1, kLado, kLado});
  const int ultima = (parar_en > 0 && parar_en < iteraciones) ? parar_en : iteraciones;
  const auto t0 = std::chrono::steady_clock::now();

  for (int it = it_inicial; it <= ultima; ++it) {
    // Lote y ruido dependen solo de (semilla, it): reanudar reproduce el run.
    const uint32_t sem_it = entrenamiento::SemillaIteracion(semilla, it);
    ManualSeed(sem_it);
    uint32_t estado = sem_it | 1u;
    opt.SetLearningRate(entrenamiento::TasaAprendizaje(it, iteraciones, calentamiento, lr, lr_min));

    for (int k = 0; k < b; ++k) {
      estado ^= estado << 13; estado ^= estado >> 17; estado ^= estado << 5;
      const int idx = static_cast<int>(estado % static_cast<uint32_t>(N));
      std::memcpy(&xb[static_cast<size_t>(k) * kPx], &x0[static_cast<size_t>(idx) * kPx], kPx * sizeof(float));
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
      const double seg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  iter %6d/%d | MSE/pixel %.5f | PSNR %6.2f dB | KL %9.1f | %.1f s\n", it,
                  iteraciones, mse, Psnr(mse), kl, seg);
    }
    if (evaluar_cada > 0 && it % evaluar_cada == 0) evaluar(it);
    if (guardar_cada > 0 && it % guardar_cada == 0 && guardar(it, archivo)) {
      std::printf("  checkpoint en la iteracion %d -> %s\n", it, archivo.c_str());
    }
    if (archivar_cada > 0 && it % archivar_cada == 0) {
      const std::string base = entrenamiento::RutaArchivada(archivo, it);
      if (guardar(it, base)) std::printf("  archivado -> %s\n", base.c_str());
    }
  }

  const bool ya_guardado = guardar_cada > 0 && ultima % guardar_cada == 0;
  if (!ya_guardado && guardar(ultima, archivo)) {
    std::printf("\n  guardado en %s (+ .dec, + .opt) tras la iteracion %d de %d\n", archivo.c_str(),
                ultima, iteraciones);
  }
  if (!(evaluar_cada > 0 && ultima % evaluar_cada == 0)) evaluar(ultima);
  std::printf("  originales y reconstrucciones -> %s\n", png.c_str());
  return 0;
}
