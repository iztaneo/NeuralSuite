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

#include <array>
#include <map>
#include <chrono>
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
  float lr = 2e-3f, beta_fin = 0.0f;   // 0 = elegir segun los pasos
  float lr_min = 0.0f, ema_decaimiento = 0.999f;
  int calentamiento = 0, guardar_cada = 0, evaluar_cada = 0, muestrear_cada = 0;
  int n_validacion = 0, parar_en = 0;
  std::string archivo = "release/unet_mnist.nsf";
  bool reanudar = false;

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
    else if (!std::strcmp(argv[i], "--beta_fin")) beta_fin = static_cast<float>(std::atof(sig("--beta_fin")));
    else if (!std::strcmp(argv[i], "--lr_min")) lr_min = static_cast<float>(std::atof(sig("--lr_min")));
    else if (!std::strcmp(argv[i], "--calentamiento")) calentamiento = std::atoi(sig("--calentamiento"));
    else if (!std::strcmp(argv[i], "--ema")) ema_decaimiento = static_cast<float>(std::atof(sig("--ema")));
    else if (!std::strcmp(argv[i], "--guardar_cada")) guardar_cada = std::atoi(sig("--guardar_cada"));
    else if (!std::strcmp(argv[i], "--evaluar_cada")) evaluar_cada = std::atoi(sig("--evaluar_cada"));
    else if (!std::strcmp(argv[i], "--muestrear_cada")) muestrear_cada = std::atoi(sig("--muestrear_cada"));
    else if (!std::strcmp(argv[i], "--n_validacion")) n_validacion = std::atoi(sig("--n_validacion"));
    else if (!std::strcmp(argv[i], "--archivo")) archivo = sig("--archivo");
    else if (!std::strcmp(argv[i], "--parar_en")) parar_en = std::atoi(sig("--parar_en"));
    else if (!std::strcmp(argv[i], "--reanudar")) reanudar = true;
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
  const int total = (n_imagenes <= 0 || n_imagenes > disponibles) ? disponibles : n_imagenes;

  // Validacion: imagenes que el optimizador NO ve. Sin esto la unica perdida
  // que se mide es la de las mismas imagenes con las que se entrena, que baja
  // aunque el modelo solo este memorizando. Con 16 imagenes memorizar es el
  // objetivo y no hace falta; con 60 000 hay que poder distinguirlo.
  //
  // Se cogen del FINAL del conjunto, no del principio: MNIST no viene ordenado
  // por clase, pero coger un bloque contiguo del principio es la clase de atajo
  // que un dia se encuentra con un conjunto que si lo esta.
  if (n_validacion < 0) n_validacion = 0;
  if (n_validacion >= total) {
    std::fprintf(stderr, "n_validacion (%d) no puede llegar a las imagenes totales (%d)\n",
                 n_validacion, total);
    return 1;
  }
  const int N = total - n_validacion;

  // MNIST llega en [0, 1]; la difusion asume datos centrados en cero, asi que
  // se pasa a [-1, 1]. Si no, la red tendria que aprender el desplazamiento y
  // el ruido gaussiano no encajaria con la varianza que el schedule supone.
  auto normalizar = [&](int desde, int cuantas) {
    Tensor t({cuantas, 1, H, W});
    for (int i = 0; i < cuantas; ++i) {
      for (int p = 0; p < PX; ++p) {
        t[static_cast<size_t>(i) * PX + p] =
            2.0f * datos.imagenes[static_cast<size_t>(desde + i) * PX + p] - 1.0f;
      }
    }
    return t;
  };
  const Tensor x0 = normalizar(0, N);
  const Tensor x0_val = normalizar(N, n_validacion);

  std::printf("============================================================\n");
  std::printf("  Difusion sobre MNIST\n");
  std::printf("  imagenes %d de %d (%d de validacion) | lote %d | iteraciones %d\n",
              N, disponibles, n_validacion, lote, iteraciones);
  std::printf("  canales %d | pasos de ruido %d | lr %.0e | semilla %d\n", canales, pasos, lr, semilla);
  if (N <= 64) std::printf("  MODO PUERTA: se busca memorizar, no generalizar\n");
  std::printf("  archivo %s%s\n", archivo.c_str(), reanudar ? " (reanudando)" : "");
  std::printf("============================================================\n");

  // Las betas del articulo (1e-4 a 0.02) estan calibradas para 1000 pasos. Con
  // menos pasos hay que subir la beta final o `x_T` no llega a ser ruido: con
  // 200 pasos y 0.02 conserva el 36% de la imagen, el modelo nunca ve ruido
  // puro y el muestreo arranca donde no sabe. Escalar por 1000/pasos mantiene
  // el ruido total acumulado.
  if (beta_fin <= 0.0f) {
    beta_fin = std::min(0.5f, 0.02f * 1000.0f / static_cast<float>(pasos));
  }

  ManualSeed(static_cast<uint32_t>(semilla));
  UNet2D unet(1, canales, dim_t, grupos);
  DiffusionSchedule schedule(pasos, 1e-4f, beta_fin);
  {
    const float resto = schedule.SenalResidual();
    std::printf("  beta final %.4f -> queda %.2f%% de la imagen en x_T  %s\n",
                beta_fin, 100.0f * resto,
                resto < 0.03f ? "ok" : "<-- DEMASIADO: x_T no es ruido puro");
  }
  AdamW opt(unet.GetParameters(), unet.GetGradients(), lr);
  EMA ema(unet.GetParameters(), ema_decaimiento);
  std::printf("  parametros: %zu | EMA %.4f\n", unet.NumParametros(), ema_decaimiento);

  // Un checkpoint es el modelo, la sombra de la EMA y el estado de Adam. Los
  // tres o ninguno: guardar solo los pesos permite muestrear pero no continuar.
  auto rutas = [&](const std::string& base) {
    return std::array<std::string, 3>{base, base + ".ema", base + ".opt"};
  };
  int it_inicial = 1;

  // Un checkpoint son tres archivos, y escribirlos en su sitio uno detras de
  // otro deja una ventana de varios segundos en la que un corte los mezcla:
  // pesos nuevos con estado de Adam viejo, por ejemplo. Los tres son
  // estructuralmente validos por separado, asi que reanudar de esa mezcla no
  // daria ningun error; solo entrenaria mal.
  //
  // Por eso se escriben primero como `.tmp` y solo se mueven a su sitio cuando
  // los tres estan completos. `std::rename` es atomico dentro del mismo sistema
  // de archivos, asi que la ventana pasa de segundos a los microsegundos entre
  // los tres renombrados. Esa ventana residual no se puede cerrar sin soporte
  // del sistema de archivos, y no se pretende: lo que la cubre es el sello.
  //
  // El sello es un `checkpoint_id` unico por llamada a `guardar()`, escrito en
  // los tres archivos. Al reanudar tienen que coincidir. Con eso, una mezcla no
  // se entrena en silencio: aborta diciendo que los archivos no son del mismo
  // checkpoint.
  auto guardar = [&](int it) {
    const auto dst = rutas(archivo);
    std::array<std::string, 3> tmp;
    for (int k = 0; k < 3; ++k) tmp[k] = dst[k] + ".tmp";

    const uint64_t id =
        (static_cast<uint64_t>(it) << 40) ^
        static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    const std::map<std::string, std::string> sello = {
        {"checkpoint_id", std::to_string(id)}, {"iteracion", std::to_string(it)}};

    bool ok = unet.GuardarPesos(tmp[0], sello);
    ema.Intercambiar();
    ok = unet.GuardarPesos(tmp[1], sello) && ok;   // la sombra, mismos nombres
    ema.Intercambiar();

    std::vector<nsf::NamedTensor> est;
    auto ms = opt.EstadoM(), vs = opt.EstadoV();
    for (size_t i = 0; i < ms.size(); ++i) {
      est.push_back({"m." + std::to_string(i), ms[i]});
      est.push_back({"v." + std::to_string(i), vs[i]});
    }
    std::map<std::string, std::string> meta_opt = sello;
    meta_opt["arch"] = "adamw";
    meta_opt["pasos_opt"] = std::to_string(opt.PasosDados());
    meta_opt["pasos_ema"] = std::to_string(ema.Pasos());
    const auto res = nsf::Save(tmp[2], est, meta_opt);
    ok = ok && res.ok;
    if (!res) std::fprintf(stderr, "  no se pudo escribir el estado del optimizador: %s\n",
                           res.error.c_str());

    if (!ok) {
      // El checkpoint anterior sigue intacto: no se ha movido nada.
      for (const std::string& f : tmp) std::remove(f.c_str());
      std::fprintf(stderr, "  checkpoint descartado; el anterior sigue en pie\n");
      return false;
    }
    for (int k = 0; k < 3; ++k) {
      if (std::rename(tmp[k].c_str(), dst[k].c_str()) != 0) {
        std::fprintf(stderr, "  no se pudo mover %s a su sitio\n", tmp[k].c_str());
        return false;
      }
    }
    return true;
  };

  if (reanudar) {
    const auto r = rutas(archivo);
    std::map<std::string, std::string> m_pesos, m_ema, m_opt;
    if (!unet.CargarPesos(r[0], &m_pesos)) {
      std::fprintf(stderr, "No se pudo reanudar desde %s\n", r[0].c_str());
      return 1;
    }
    // La sombra se lee cargandola en el modelo y sacandola con el intercambio,
    // que reutiliza la comprobacion de nombres en vez de leer el archivo a pelo.
    ema.Intercambiar();
    if (!unet.CargarPesos(r[1], &m_ema)) {
      std::fprintf(stderr, "No se pudo leer la EMA de %s\n", r[1].c_str());
      return 1;
    }
    ema.Intercambiar();

    std::vector<nsf::NamedTensor> est;
    auto ms = opt.EstadoM(), vs = opt.EstadoV();
    for (size_t i = 0; i < ms.size(); ++i) {
      est.push_back({"m." + std::to_string(i), ms[i]});
      est.push_back({"v." + std::to_string(i), vs[i]});
    }
    const auto res = nsf::Load(r[2], est, {{"arch", "adamw"}}, &m_opt);
    if (!res) {
      std::fprintf(stderr, "No se pudo leer el estado del optimizador: %s\n", res.error.c_str());
      return 1;
    }

    // El sello. Sin esto, una mezcla de dos checkpoints se cargaria sin
    // protestar: los tres archivos son validos por separado.
    const std::string id = m_pesos["checkpoint_id"];
    if (id.empty() || m_ema["checkpoint_id"] != id || m_opt["checkpoint_id"] != id) {
      std::fprintf(stderr,
                   "Los tres archivos no son del mismo checkpoint (pesos '%s', "
                   "EMA '%s', optimizador '%s'). Probablemente un corte durante "
                   "el guardado; usa un checkpoint anterior.\n",
                   id.c_str(), m_ema["checkpoint_id"].c_str(),
                   m_opt["checkpoint_id"].c_str());
      return 1;
    }

    opt.FijarPasosDados(std::stoi(m_opt["pasos_opt"]));
    ema.FijarPasos(std::stoi(m_opt["pasos_ema"]));
    it_inicial = std::stoi(m_opt["iteracion"]) + 1;
    std::printf("  reanudado en la iteracion %d (Adam %d pasos, EMA %d)\n",
                it_inicial, opt.PasosDados(), ema.Pasos());
  }
  std::printf("\n");

  Rng rng{static_cast<uint32_t>(semilla) * 2654435761u + 1u};
  const int lote_real = std::min(lote, N);

  Tensor xb({lote_real, 1, H, W}), eps({lote_real, 1, H, W}), xt, t({lote_real});
  Tensor dout({lote_real, 1, H, W});
  const size_t elems = xb.TotalSize();
  const auto t0 = std::chrono::steady_clock::now();

  // Franjas de t, para no promediar dos regimenes distintos en un solo numero.
  double sum_baja = 0, sum_alta = 0; int n_baja = 0, n_alta = 0;

  // Perdida sobre un conjunto dado, con t barrido de forma determinista y
  // ruido fijo: si el ruido cambiara entre evaluaciones, la curva se movería
  // por el sorteo y no por el modelo, y no se podrian comparar dos momentos.
  auto evaluar = [&](const Tensor& conjunto, int cuantas, uint32_t sem) {
    if (cuantas == 0) return -1.0;
    Rng r2{sem * 2654435761u + 7u};
    Tensor xe({lote_real, 1, H, W}), ee({lote_real, 1, H, W}), xte, te({lote_real});
    double acc = 0.0; int veces = 0;
    ManualSeed(sem);
    for (int rep = 0; rep < 16; ++rep) {
      for (int b = 0; b < lote_real; ++b) {
        std::memcpy(&xe[static_cast<size_t>(b) * PX],
                    &conjunto[static_cast<size_t>(r2.Entero(cuantas)) * PX],
                    static_cast<size_t>(PX) * sizeof(float));
        te[b] = static_cast<float>(r2.Entero(pasos));
      }
      ee.RandomNormal(0.0f, 1.0f);
      schedule.QSample(xe, ee, te, &xte);
      const Tensor pr = unet.Forward(xte, te);
      double l = 0.0;
      for (size_t i = 0; i < elems; ++i) { const double d = pr[i] - ee[i]; l += d * d; }
      acc += l / static_cast<double>(elems); ++veces;
    }
    return acc / veces;
  };

  // `--parar_en` corta antes sin tocar el plan: el coseno del learning rate
  // sigue calculandose sobre `iteraciones`, asi que parar y reanudar da el
  // mismo entrenamiento que no parar. Bajar `--iteraciones` en su lugar seria
  // otro entrenamiento distinto, con el decaimiento comprimido.
  const int ultima = (parar_en > 0 && parar_en < iteraciones) ? parar_en : iteraciones;

  for (int it = it_inicial; it <= ultima; ++it) {
    // Cada iteracion se siembra en funcion de (semilla, it), no del estado que
    // arrastre el generador. Asi la iteracion n usa el mismo lote y el mismo
    // ruido tanto si se llega de un tiron como reanudando, que es lo unico que
    // hace que un checkpoint sea equivalente a no haber parado. La alternativa
    // —guardar el estado del mt19937 global y el del generador de lotes— exige
    // acordarse de los dos, y ademas la evaluacion y el muestreo periodicos
    // tocan el global por el camino.
    const uint32_t sem_it = (static_cast<uint32_t>(semilla) * 0x9E3779B1u) ^
                            (static_cast<uint32_t>(it) * 0x85EBCA6Bu);
    ManualSeed(sem_it);
    rng.s = sem_it | 1u;   // xorshift no admite el cero

    // Calentamiento lineal y despues coseno hasta `lr_min`. El calentamiento no
    // es cosmetico: Adam arranca con m = v = 0 y la correccion de sesgo hace
    // que los primeros pasos sean del tamano maximo, justo cuando los pesos son
    // aleatorios y el gradiente no apunta a nada util.
    float lr_ahora = lr;
    if (calentamiento > 0 && it <= calentamiento) {
      lr_ahora = lr * static_cast<float>(it) / static_cast<float>(calentamiento);
    } else if (iteraciones > calentamiento) {
      const float avance = static_cast<float>(it - calentamiento) /
                           static_cast<float>(iteraciones - calentamiento);
      lr_ahora = lr_min + 0.5f * (lr - lr_min) * (1.0f + std::cos(3.14159265f * avance));
    }
    opt.SetLearningRate(lr_ahora);

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
    //
    // La perdida se acumula POR EJEMPLO y no solo en total, porque las franjas
    // de t se reparten por ejemplo. La primera version sumaba la perdida del
    // lote entero a la franja de cada muestra: entonces `t bajo` y `t alto`
    // eran las dos la misma cifra —la del lote— pesada por cuantas muestras de
    // cada tipo tenia cada lote, asi que se movian juntas y las diferencias que
    // mostraban venian de la composicion de los lotes, no de la dificultad de
    // cada franja. Justo lo contrario de lo que la columna decia medir.
    double perdida = 0.0;
    for (int b = 0; b < lote_real; ++b) {
      double l_b = 0.0;
      const size_t base = static_cast<size_t>(b) * PX;
      for (int q = 0; q < PX; ++q) {
        const double d = static_cast<double>(pred[base + q]) - eps[base + q];
        l_b += d * d;
        dout[base + q] = static_cast<float>(2.0 * d / static_cast<double>(elems));
      }
      l_b /= static_cast<double>(PX);
      perdida += l_b;
      if (t[b] < pasos / 2) { sum_baja += l_b; ++n_baja; }
      else { sum_alta += l_b; ++n_alta; }
    }
    perdida /= static_cast<double>(lote_real);

    opt.ZeroGrad();
    unet.Backward(dout);
    opt.Step();
    ema.Actualizar();

    if (it % reportar_cada == 0 || it == 1) {
      const double seg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  iter %5d/%d | perdida %.4f | t bajo %.4f | t alto %.4f | %.1f s\n",
                  it, iteraciones, perdida,
                  n_baja ? sum_baja / n_baja : 0.0, n_alta ? sum_alta / n_alta : 0.0, seg);
      sum_baja = sum_alta = 0; n_baja = n_alta = 0;
    }

    // 5. Validacion: la unica curva que distingue aprender de memorizar.
    if (evaluar_cada > 0 && it % evaluar_cada == 0) {
      const double l_tr = evaluar(x0, N, 4242u);
      const double l_va = evaluar(x0_val, n_validacion, 4242u);
      if (l_va >= 0.0) {
        std::printf("  eval %5d | entren. %.4f | valid. %.4f | brecha %+.4f\n",
                    it, l_tr, l_va, l_va - l_tr);
      } else {
        std::printf("  eval %5d | entren. %.4f | (sin validacion)\n", it, l_tr);
      }
    }

    // 6. Muestrear con los pesos de la EMA, que son con los que se muestrea de
    //    verdad. Ver el progreso en imagenes durante el entrenamiento cuesta
    //    unos segundos y evita descubrir al final que no iba a ninguna parte.
    if (muestrear_cada > 0 && it % muestrear_cada == 0) {
      ema.Intercambiar();
      Predictor red = [&](const Tensor& xx, const Tensor& tt) { return unet.Forward(xx, tt); };
      DDIMSampler vistazo(schedule, std::max(2, pasos / 10), 0.0f);
      vistazo.RecortarX0(true);
      Tensor r1({1, 1, H, W});
      ManualSeed(static_cast<uint32_t>(semilla) + 5000u);
      r1.RandomNormal(0.0f, 1.0f);
      const Tensor y = vistazo.Muestrear(red, r1, RuidoNulo());
      std::printf("  muestra en la iteracion %d (EMA, DDIM):\n", it);
      Dibujar(y, 0, H, W);
      ema.Intercambiar();
    }

    // 2. Checkpoint. Sin esto, un run de horas que se corte no deja nada.
    if (guardar_cada > 0 && it % guardar_cada == 0) {
      if (guardar(it)) std::printf("  checkpoint en la iteracion %d -> %s\n", it, archivo.c_str());
    }
  }

  // Si la ultima iteracion ya cayo en un multiplo de `guardar_cada`, el
  // checkpoint acaba de escribirse: repetirlo cuesta un segundo y, sobre todo,
  // hace dudar de si son dos cosas distintas al leer el log.
  const bool ya_guardado = guardar_cada > 0 && ultima % guardar_cada == 0;
  if (!ya_guardado) {
    if (guardar(ultima)) {
      std::printf("\n  guardado en %s (+ .ema, +.opt) tras la iteracion %d de %d\n",
                  archivo.c_str(), ultima, iteraciones);
    }
  }

  // A partir de aqui se mide y se muestrea con la EMA, que es lo que se usaria
  // en produccion. Medir con los pesos vivos y muestrear con los promediados
  // seria informar de dos modelos distintos como si fueran uno.
  ema.Intercambiar();

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
  // Diagnostico: calidad de x_0 predicho en funcion de t. Es la magnitud que
  // gobierna la generacion, y NO es la perdida. La perdida mide el error en
  // `eps`, y con t grande `eps ~ x_t`, asi que una red que solo copiase su
  // entrada ya sacaria perdida baja; pero al despejar x_0 se divide por
  // sqrt(ab[t]), que ahi es diminuto, y ese error pequeno se amplifica. Si x_0
  // predicho es basura con t grande, el muestreo no puede funcionar por bien
  // que este el bucle: arranca justo en ese regimen.
  {
    std::printf("\n  --- calidad de x_0 predicho segun t (0 = perfecto) ---\n");
    std::printf("  %5s | %10s | %10s | %s\n", "t", "err x_0", "err eps", "copiar la entrada daria");
    Tensor u({1, 1, H, W}), rr({1, 1, H, W}), tt({1}), xtt, x0p;
    for (int tv : {5, 40, 80, 120, 160, 195}) {
      double acc_x0 = 0.0, acc_eps = 0.0, acc_copia = 0.0;
      const int reps = std::min(N, 8);
      for (int i = 0; i < reps; ++i) {
        std::memcpy(&u[0], &x0[static_cast<size_t>(i) * PX], static_cast<size_t>(PX) * sizeof(float));
        rr.RandomNormal(0.0f, 1.0f);
        tt[0] = static_cast<float>(tv);
        schedule.QSample(u, rr, tt, &xtt);
        const Tensor pr = unet.Forward(xtt, tt);
        schedule.PredecirX0(xtt, pr, tt, &x0p);
        double e0 = 0.0, ee = 0.0, ec = 0.0;
        for (int q = 0; q < PX; ++q) {
          const double a = x0p[q] - u[q];        e0 += a * a;
          const double b = pr[q] - rr[q];        ee += b * b;
          const double c = xtt[q] - rr[q];       ec += c * c;   // si copiara x_t
        }
        acc_x0 += std::sqrt(e0 / PX); acc_eps += ee / PX; acc_copia += ec / PX;
      }
      std::printf("  %5d | %10.4f | %10.4f | %10.4f\n", tv, acc_x0 / reps,
                  acc_eps / reps, acc_copia / reps);
    }
  }

  // --- Muestreo desde ruido puro.
  //
  // Esto es lo que ninguna perdida demuestra. Con 16 imagenes memorizadas, un
  // muestreador correcto tiene que devolver **una de esas 16**; si sale ruido,
  // o una mancha que no es ningun digito, el fallo esta en el bucle de muestreo
  // y no en el entrenamiento. Por eso se compara cada muestra contra las N
  // imagenes del conjunto y se dice a cual se parece y cuanto: mirar el dibujo
  // y decir "parece un cinco" no es una medida.
  {
    const int n_muestras = std::min(4, N);
    Tensor ruido({n_muestras, 1, H, W});
    ManualSeed(static_cast<uint32_t>(semilla) + 1000u);
    ruido.RandomNormal(0.0f, 1.0f);
    Predictor red = [&](const Tensor& xx, const Tensor& tt) { return unet.Forward(xx, tt); };

    auto informar = [&](const char* nombre, const Tensor& y, double seg) {
      std::printf("\n  --- %s (%.1f s) ---\n", nombre, seg);
      for (int b = 0; b < n_muestras; ++b) {
        // Distancia a la imagen mas cercana del conjunto, normalizada por la
        // distancia media: 0 seria una copia exacta, y ~1 quiere decir que no
        // se parece a ninguna mas que cualquier par al azar.
        double mejor = 1e30; int cual = -1, suma_n = 0; double suma = 0.0;
        for (int i = 0; i < N; ++i) {
          double d = 0.0;
          for (int p = 0; p < PX; ++p) {
            const double e = y[static_cast<size_t>(b) * PX + p] -
                             x0[static_cast<size_t>(i) * PX + p];
            d += e * e;
          }
          d = std::sqrt(d / PX);
          suma += d; ++suma_n;
          if (d < mejor) { mejor = d; cual = i; }
        }
        std::printf("  muestra %d -> mas parecida a la imagen %2d | distancia %.4f "
                    "(media a todas %.4f)\n", b, cual, mejor, suma / suma_n);
      }
      std::printf("\n");
      Dibujar(y, 0, H, W);
    };

    DDPMSampler ddpm(schedule);
    ddpm.RecortarX0(true);
    auto c0 = std::chrono::steady_clock::now();
    const Tensor y_ddpm = ddpm.Muestrear(red, ruido, RuidoGaussiano(static_cast<uint32_t>(semilla)));
    informar("DDPM, todos los pasos", y_ddpm,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count());

    const int np = std::max(2, pasos / 10);
    DDIMSampler ddim(schedule, np, 0.0f);
    ddim.RecortarX0(true);
    c0 = std::chrono::steady_clock::now();
    const Tensor y_ddim = ddim.Muestrear(red, ruido, RuidoNulo());
    char etiqueta[80];
    std::snprintf(etiqueta, sizeof(etiqueta), "DDIM, %d pasos de %d (eta=0)", np, pasos);
    informar(etiqueta, y_ddim,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count());
  }

  std::printf("\n");
  return 0;
}
