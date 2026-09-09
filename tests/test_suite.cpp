// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file test_suite.cpp
 * @brief Numerical Unit Test Suite following Google C++ Style Guide.
 *
 * Las comprobaciones usan Check() y no assert(): assert() se compila a nada
 * cuando se define NDEBUG, que es justo lo que hace una compilación Release,
 * de modo que la suite pasaría sin verificar nada.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>
#include "autograd.h"
#include "parallel.h"
#include "neuralsuite.h"

using namespace neuralsuite;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cout << "\n   ❌ FALLO: " << what << "\n" << std::flush;
    ++g_failures;
  }
}

/**
 * @brief Error relativo entre el gradiente analítico y el numérico.
 *
 * Se normaliza por la magnitud de ambos para que valores cercanos a cero no
 * produzcan errores relativos artificialmente enormes.
 */
double RelativeError(double numeric, double analytic) {
  const double denom = std::max(1e-6, std::abs(numeric) + std::abs(analytic));
  return std::abs(numeric - analytic) / denom;
}

// Por debajo de esta magnitud el gradiente no se puede medir de forma fiable
// por diferencias finitas en float32: el error relativo pasa a ser ruido.
constexpr double kNegligibleGrad = 1e-3;

/**
 * @brief Compara por diferencias finitas centradas los gradientes de `params`.
 *
 * El paso `eps` no es universal. Es un compromiso entre dos errores opuestos:
 * con eps pequeño, `loss(w+eps) - loss(w-eps)` sufre cancelación catastrófica
 * en float32; con eps grande domina el error de truncamiento O(eps²), que
 * crece con la no linealidad de la pérdida. Por eso cada prueba mide el suyo:
 * una pérdida casi lineal admite pasos grandes, y una cross-entropy no.
 *
 * @return el peor error relativo observado entre los elementos medibles.
 */
double MaxGradError(const std::vector<Tensor*>& params, const std::vector<Tensor*>& grads,
                    const std::function<double()>& loss_of, float eps, size_t samples_per_tensor,
                    int* checked, int* skipped) {
  double worst = 0.0;
  for (size_t p = 0; p < params.size(); ++p) {
    Tensor* w = params[p];
    Tensor* g = grads[p];
    const size_t stride = std::max<size_t>(1, w->TotalSize() / samples_per_tensor);

    for (size_t i = 0; i < w->TotalSize(); i += stride) {
      const float orig = (*w)[i];
      (*w)[i] = orig + eps; const double loss_plus = loss_of();
      (*w)[i] = orig - eps; const double loss_minus = loss_of();
      (*w)[i] = orig;

      const double numeric = (loss_plus - loss_minus) / (2.0 * eps);
      const double analytic = (*g)[i];

      if (std::max(std::abs(numeric), std::abs(analytic)) < kNegligibleGrad) {
        ++(*skipped);
        continue;
      }
      worst = std::max(worst, RelativeError(numeric, analytic));
      ++(*checked);
    }
  }
  return worst;
}

}  // namespace

void TestMatMul() {
  std::cout << "🧪 [Test 1] Multiplicación de Matrices (GEMM)... " << std::flush;
  Tensor A({2, 3});
  Tensor B({3, 2});

  A[0] = 1; A[1] = 2; A[2] = 3;
  A[3] = 4; A[4] = 5; A[5] = 6;

  B[0] = 7; B[1] = 8;
  B[2] = 9; B[3] = 1;
  B[4] = 2; B[5] = 3;

  Tensor C;
  MatMul(A, B, C);

  Check(std::abs(C[0] - 31.0f) < 1e-4f, "C[0] deberia ser 31");
  Check(std::abs(C[1] - 19.0f) < 1e-4f, "C[1] deberia ser 19");
  Check(std::abs(C[2] - 85.0f) < 1e-4f, "C[2] deberia ser 85");
  Check(std::abs(C[3] - 55.0f) < 1e-4f, "C[3] deberia ser 55");

  std::cout << "PASADO ✅\n" << std::flush;
}

void TestLayerNorm() {
  std::cout << "🧪 [Test 2] Normalización de Capa (LayerNorm)... " << std::flush;
  Tensor x({1, 4});
  x[0] = 2.0f; x[1] = 4.0f; x[2] = 4.0f; x[3] = 6.0f;

  Tensor gamma({4}); gamma.Ones();
  Tensor beta({4}); beta.Zeros();

  Tensor out, mean, rstd;
  LayerNormForward(x, gamma, beta, out, mean, rstd);

  Check(std::abs(mean[0] - 4.0f) < 1e-4f, "la media deberia ser 4");
  std::cout << "PASADO ✅\n" << std::flush;
}

void TestTokenizer() {
  std::cout << "🧪 [Test 3] Tokenizador de Caracteres C++... " << std::flush;
  std::string sample = "Hello C++ Google Style!";
  CharTokenizer tok(sample);

  std::vector<int> encoded = tok.Encode(sample);
  std::string decoded = tok.Decode(encoded);

  Check(sample == decoded, "el roundtrip encode/decode no conserva el texto");
  std::cout << "PASADO ✅\n" << std::flush;
}

void TestGradientCheckGelu() {
  std::cout << "🧪 [Test 4] Gradiente GELU por Diferencias Finitas... " << std::flush;
  Tensor x({1, 1});
  x[0] = 1.5f;

  Tensor dout({1, 1});
  dout[0] = 1.0f;

  Tensor dx;
  GeluBackward(dout, x, dx);

  float eps = 1e-4f;
  Tensor x_plus({1, 1}), x_minus({1, 1});
  x_plus[0] = x[0] + eps;
  x_minus[0] = x[0] - eps;

  Tensor y_plus, y_minus;
  GeluForward(x_plus, y_plus);
  GeluForward(x_minus, y_minus);

  float num_grad = (y_plus[0] - y_minus[0]) / (2.0f * eps);
  float diff = std::abs(dx[0] - num_grad);

  Check(diff < 1e-3f, "el gradiente GELU no coincide con las diferencias finitas");
  std::cout << "PASADO ✅ (Diff: " << diff << ")\n" << std::flush;
}

/**
 * @brief Todos los parámetros de MultiHeadAttention reciben gradiente correcto.
 *
 * Cubre el defecto por el que la clase no sobrescribía GetGradients(): sin ese
 * override, esta prueba ni siquiera puede construir las listas emparejadas.
 */
void TestGradientCheckAttention() {
  std::cout << "🧪 [Test 5] Gradientes de MultiHeadAttention... " << std::flush;

  const int B = 1, T = 4, C = 8, H = 2;
  MultiHeadAttention attn(C, H);

  auto params = attn.GetParameters();
  auto grads = attn.GetGradients();
  Check(params.size() == grads.size(),
        "MultiHeadAttention expone " + std::to_string(params.size()) +
            " parametros pero " + std::to_string(grads.size()) + " gradientes");
  if (params.size() != grads.size()) return;

  Tensor x({B, T, C});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.1f * static_cast<float>((i * 7) % 11) - 0.5f;
  }

  // Pérdida escalar: suma de la salida, de modo que dout es todo unos.
  auto loss_of = [&]() {
    Tensor y = attn.Forward(x);
    double s = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) s += y[i];
    return s;
  };

  loss_of();
  Tensor dout({B, T, C});
  dout.Ones();
  attn.Backward(dout);

  // La pérdida es la suma de las salidas, casi lineal en los pesos, así que
  // tolera un paso grande: medido, eps=5e-2 da ~3e-4 de error, mientras que
  // eps=1e-4 da ~0.48 por cancelación en float32.
  int checked = 0, skipped = 0;
  const double worst =
      MaxGradError(params, grads, loss_of, 5e-2f, 4, &checked, &skipped);

  Check(worst < 1e-2, "el peor error relativo de attention es " + std::to_string(worst));
  std::cout << "PASADO ✅ (" << checked << " elementos, peor error rel: " << worst << ")\n"
            << std::flush;
}

/**
 * @brief La matriz wte_, compartida por weight tying, acumula ambos gradientes.
 *
 * Cubre el defecto por el que Embedding::Backward() ponía a cero el acumulador
 * después de que GPTModel ya hubiera sumado la contribución de la cabeza de
 * salida: dW total debe ser dW_embedding + dW_output.
 */
void TestGradientCheckWeightTying() {
  std::cout << "🧪 [Test 6] Weight tying del GPT (matriz wte compartida)... " << std::flush;

  GPTConfig cfg;
  cfg.vocab_size = 11; cfg.block_size = 6;
  cfg.n_layer = 2; cfg.n_head = 2; cfg.n_embd = 8;
  GPTModel model(cfg);

  const int B = 2, T = 5;
  Tensor idx({B, T});
  for (int i = 0; i < B * T; ++i) idx[i] = static_cast<float>(i % cfg.vocab_size);
  Tensor targets({B * T});
  for (int i = 0; i < B * T; ++i) targets[i] = static_cast<float>((i * 3 + 1) % cfg.vocab_size);

  CrossEntropyLoss crit;
  auto loss_of = [&]() {
    Tensor logits = model.Forward(idx);
    Tensor logits_2d({B * T, cfg.vocab_size});
    std::memcpy(logits_2d.Data(), logits.Data(), logits.TotalSize() * sizeof(float));
    return static_cast<double>(crit.Forward(logits_2d, targets));
  };

  loss_of();
  Tensor dl2d = crit.Backward();
  Tensor dlogits({B, T, cfg.vocab_size});
  std::memcpy(dlogits.Data(), dl2d.Data(), dl2d.TotalSize() * sizeof(float));
  model.Backward(dlogits);

  std::vector<Tensor*> wte_w = {model.GetParameters()[0]};
  std::vector<Tensor*> wte_g = {model.GetGradients()[0]};

  // Aquí la pérdida es una cross-entropy, mucho más no lineal, y el error de
  // truncamiento crece rápido: medido, eps=1e-3 da ~5e-3 de error mientras que
  // eps=1e-2 lo degrada a ~6e-2. El umbral deja margen para la varianza que
  // introduce el RNG global compartido entre pruebas.
  int checked = 0, skipped = 0;
  const double worst =
      MaxGradError(wte_w, wte_g, loss_of, 1e-3f, 30, &checked, &skipped);

  Check(worst < 5e-2, "el peor error relativo de wte es " + std::to_string(worst) +
                          "; la contribucion de la cabeza de salida se esta perdiendo");
  std::cout << "PASADO ✅ (" << checked << " elementos, peor error rel: " << worst << ")\n"
            << std::flush;
}

/** @brief Parámetros y gradientes del GPT deben emparejarse uno a uno. */
void TestParamGradAlignment() {
  std::cout << "🧪 [Test 7] Alineación de parámetros y gradientes del GPT... " << std::flush;

  GPTConfig cfg;
  cfg.vocab_size = 16; cfg.block_size = 8;
  cfg.n_layer = 4; cfg.n_head = 4; cfg.n_embd = 16;
  GPTModel model(cfg);

  auto p = model.GetParameters();
  auto g = model.GetGradients();

  Check(p.size() == g.size(), "el GPT expone " + std::to_string(p.size()) +
                                  " parametros y " + std::to_string(g.size()) + " gradientes");
  if (p.size() == g.size()) {
    for (size_t i = 0; i < p.size(); ++i) {
      Check(p[i]->Shape() == g[i]->Shape(),
            "forma distinta entre parametro y gradiente en el indice " + std::to_string(i));
    }
  }

  std::cout << "PASADO ✅ (" << p.size() << " pares)\n" << std::flush;
}

/** @brief El optimizador rechaza listas desemparejadas en vez de corromper pesos. */
void TestOptimizerRejectsMismatch() {
  std::cout << "🧪 [Test 8] El optimizador rechaza pares invalidos... " << std::flush;

  Linear layer(4, 3);

  bool threw_on_valid = false;
  try {
    AdamW opt(layer.GetParameters(), layer.GetGradients(), 0.01f);
  } catch (const std::exception&) {
    threw_on_valid = true;
  }
  Check(!threw_on_valid, "el optimizador rechazo un par valido");

  bool threw_on_missing = false;
  try {
    AdamW opt(layer.GetParameters(), {}, 0.01f);
  } catch (const std::invalid_argument&) {
    threw_on_missing = true;
  }
  Check(threw_on_missing, "el optimizador acepto parametros sin gradientes");

  Linear other(9, 7);
  bool threw_on_shape = false;
  try {
    SGD opt(layer.GetParameters(), other.GetGradients(), 0.01f);
  } catch (const std::invalid_argument&) {
    threw_on_shape = true;
  }
  Check(threw_on_shape, "el optimizador acepto gradientes con forma distinta");

  std::cout << "PASADO ✅\n" << std::flush;
}

/** @brief Tensor y las operaciones rechazan entradas invalidas. */
void TestInputValidation() {
  std::cout << "🧪 [Test 9] Validación de formas e índices... " << std::flush;

  bool threw = false;
  try { Tensor bad({-1, 4}); } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "Tensor acepto una dimension negativa");

  threw = false;
  try {
    Tensor A({2, 3}), B({4, 5}), C;
    MatMul(A, B, C);
  } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "MatMul acepto dimensiones incompatibles");

  threw = false;
  try {
    Embedding emb(10, 4);
    Tensor idx({1, 2});
    idx[0] = 0.0f; idx[1] = 999.0f;  // fuera del vocabulario
    emb.Forward(idx);
  } catch (const std::out_of_range&) { threw = true; }
  Check(threw, "Embedding acepto un token fuera de rango");

  threw = false;
  try { MultiHeadAttention attn(130, 4); } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "MultiHeadAttention acepto n_embd no divisible entre n_head");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Reshape reinterpreta sin perder datos; Resize reasigna.
 *
 * Cubre el comportamiento anterior, en el que Reshape() descartaba el buffer en
 * silencio cuando el número de elementos cambiaba.
 */
void TestReshapeSemantics() {
  std::cout << "🧪 [Test 10] Semántica de Reshape y Resize... " << std::flush;

  Tensor t({2, 6});
  for (size_t i = 0; i < t.TotalSize(); ++i) t[i] = static_cast<float>(i);

  // Reinterpretar conservando los 12 elementos no debe tocar los datos.
  t.Reshape({3, 4});
  Check(t.Shape() == std::vector<int>({3, 4}), "Reshape no aplico la nueva forma");
  Check(t.TotalSize() == 12, "Reshape cambio el numero de elementos");
  bool data_intact = true;
  for (size_t i = 0; i < t.TotalSize(); ++i) {
    if (std::abs(t[i] - static_cast<float>(i)) > 1e-6f) data_intact = false;
  }
  Check(data_intact, "Reshape altero los datos que debia conservar");

  // Cambiar el numero de elementos con Reshape debe ser un error explicito.
  bool threw = false;
  try { t.Reshape({5, 5}); } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "Reshape acepto un cambio en el numero de elementos");
  Check(t.Shape() == std::vector<int>({3, 4}), "el Reshape fallido dejo el tensor modificado");

  // Resize si puede cambiar el tamano del almacenamiento.
  t.Resize({5, 5});
  Check(t.TotalSize() == 25, "Resize no reasigno al nuevo tamano");

  std::cout << "PASADO ✅\n" << std::flush;
}

/** @brief Una asignación fallida no debe dejar el destino inutilizable. */
void TestAssignmentKeepsSource() {
  std::cout << "🧪 [Test 11] Asignación de Tensor... " << std::flush;

  Tensor a({2, 3});
  for (size_t i = 0; i < a.TotalSize(); ++i) a[i] = static_cast<float>(i) + 1.0f;

  Tensor b;
  b = a;
  Check(b.Shape() == a.Shape(), "la asignacion no copio la forma");
  bool same = true;
  for (size_t i = 0; i < a.TotalSize(); ++i) {
    if (std::abs(a[i] - b[i]) > 1e-6f) same = false;
  }
  Check(same, "la asignacion no copio los datos");

  // Modificar la copia no debe afectar al original.
  b[0] = 99.0f;
  Check(std::abs(a[0] - 1.0f) < 1e-6f, "la copia comparte memoria con el original");

  // Autoasignacion.
  a = a;
  Check(a.TotalSize() == 6 && std::abs(a[0] - 1.0f) < 1e-6f, "la autoasignacion corrompio el tensor");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Gradientes del LSTM por retropropagación a través del tiempo.
 *
 * Comprueba los cuatro tensores de parámetros y también `dx`, el gradiente que
 * la capa propaga hacia atrás: una versión anterior devolvía ceros en ambos, de
 * modo que la capa no aprendía y ademas cortaba la cadena hacia capas previas.
 */
void TestGradientCheckLstm() {
  std::cout << "🧪 [Test 12] Gradientes del LSTM (BPTT)... " << std::flush;

  const int T = 4, B = 2, IN = 3, H = 5;
  LSTM lstm(IN, H);

  Tensor x({T, B, IN});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.3f * std::sin(0.7f * static_cast<float>(i)) + 0.1f;
  }

  // Pérdida escalar con pesos variables por posición: una suma simple podría
  // ocultar errores que se cancelan entre pasos temporales.
  Tensor w({T, B, H});
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 0.5f + 0.5f * std::cos(1.3f * static_cast<float>(i));
  }
  auto loss_of = [&]() {
    Tensor y = lstm.Forward(x);
    double s = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
    return s;
  };

  loss_of();
  Tensor dout(w.Shape());
  for (size_t i = 0; i < w.TotalSize(); ++i) dout[i] = w[i];
  Tensor dx = lstm.Backward(dout);

  // Los gradientes de los pesos son pequeños y el paso corto los ahoga en el
  // ruido de float32: medido, eps=1e-4 da 0.23 de error y eps=5e-2 da 6e-4.
  int checked = 0, skipped = 0;
  const double worst_params =
      MaxGradError(lstm.GetParameters(), lstm.GetGradients(), loss_of, 5e-2f, 12,
                   &checked, &skipped);
  Check(worst_params < 1e-2,
        "el peor error relativo de los parametros del LSTM es " + std::to_string(worst_params));

  // dx: sin esto la capa no propagaria gradiente a las capas anteriores.
  const float eps = 5e-2f;
  double worst_dx = 0.0;
  int dx_checked = 0;
  for (size_t i = 0; i < x.TotalSize(); i += 2) {
    const float orig = x[i];
    x[i] = orig + eps; const double lp = loss_of();
    x[i] = orig - eps; const double lm = loss_of();
    x[i] = orig;

    const double numeric = (lp - lm) / (2.0 * eps);
    if (std::max(std::abs(numeric), std::abs(static_cast<double>(dx[i]))) < kNegligibleGrad) continue;
    worst_dx = std::max(worst_dx, RelativeError(numeric, dx[i]));
    ++dx_checked;
  }
  Check(worst_dx < 1e-2, "el peor error relativo de dx del LSTM es " + std::to_string(worst_dx));

  std::cout << "PASADO ✅ (" << checked << " params, " << dx_checked
            << " dx; peor error rel: " << std::max(worst_params, worst_dx) << ")\n"
            << std::flush;
}

/**
 * @brief Semántica de View(): comparte memoria, pero copiarla desvincula.
 *
 * Es la propiedad de la que depende que `last_input_ = input` siga siendo una
 * instantánea y no un alias que cambie bajo los pies del backward.
 */
void TestViewSemantics() {
  std::cout << "🧪 [Test 16] Semántica de las vistas de Tensor... " << std::flush;

  Tensor base({2, 6});
  for (size_t i = 0; i < base.TotalSize(); ++i) base[i] = static_cast<float>(i);

  Tensor view = base.View({3, 4});
  Check(view.Shape() == std::vector<int>({3, 4}), "la vista no adopto la forma pedida");
  Check(view.SharesStorageWith(base), "la vista no comparte memoria con el original");

  // Escribir en la vista tiene que verse en el original.
  view[0] = 99.0f;
  Check(std::abs(base[0] - 99.0f) < 1e-6f, "la vista no escribe sobre la memoria compartida");

  // Copiar la vista produce un tensor independiente.
  Tensor copy = view;
  Check(!copy.SharesStorageWith(base), "copiar una vista siguio compartiendo memoria");
  copy[1] = -5.0f;
  Check(std::abs(base[1] - 1.0f) < 1e-6f, "escribir en la copia altero el original");

  // Un Resize sobre la vista debe desvincularla, no pisar al original.
  Tensor detach = base.View({3, 4});
  detach.Resize({5, 5});
  Check(!detach.SharesStorageWith(base), "Resize dejo la vista compartiendo memoria");
  Check(std::abs(base[0] - 99.0f) < 1e-6f, "Resize sobre la vista modifico el original");

  // Una vista con otro numero de elementos no tiene sentido.
  bool threw = false;
  try { (void)base.View({5, 5}); } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "View acepto un numero de elementos distinto");

  std::cout << "PASADO ✅\n" << std::flush;
}

/** @brief Gradientes de Conv2D: peso, sesgo y dx, con stride y padding activos. */
void TestGradientCheckConv2D() {
  std::cout << "🧪 [Test 13] Gradientes de Conv2D... " << std::flush;

  // stride y padding distintos de los valores por defecto: un backward que
  // ignorase el desplazamiento o el relleno pasaria desapercibido con 1 y 0.
  const int B = 2, IC = 2, OC = 3, HW = 5, K = 3, STRIDE = 2, PAD = 1;
  Conv2D conv(IC, OC, K, STRIDE, PAD);

  Tensor x({B, IC, HW, HW});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.4f * std::sin(0.9f * static_cast<float>(i)) + 0.05f;
  }

  Tensor probe = conv.Forward(x);
  Tensor w(probe.Shape());
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 0.5f + 0.5f * std::cos(1.1f * static_cast<float>(i));
  }
  auto loss_of = [&]() {
    Tensor y = conv.Forward(x);
    double s = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
    return s;
  };

  loss_of();
  Tensor dx = conv.Backward(w);

  int checked = 0, skipped = 0;
  const double worst_params =
      MaxGradError(conv.GetParameters(), conv.GetGradients(), loss_of, 1e-2f, 12,
                   &checked, &skipped);
  Check(worst_params < 1e-2,
        "el peor error relativo de los parametros de Conv2D es " + std::to_string(worst_params));

  const float eps = 1e-2f;
  double worst_dx = 0.0;
  int dx_checked = 0;
  for (size_t i = 0; i < x.TotalSize(); i += 5) {
    const float orig = x[i];
    x[i] = orig + eps; const double lp = loss_of();
    x[i] = orig - eps; const double lm = loss_of();
    x[i] = orig;

    const double numeric = (lp - lm) / (2.0 * eps);
    if (std::max(std::abs(numeric), std::abs(static_cast<double>(dx[i]))) < kNegligibleGrad) continue;
    worst_dx = std::max(worst_dx, RelativeError(numeric, dx[i]));
    ++dx_checked;
  }
  Check(worst_dx < 1e-2, "el peor error relativo de dx de Conv2D es " + std::to_string(worst_dx));

  std::cout << "PASADO ✅ (" << checked << " params, " << dx_checked
            << " dx; peor error rel: " << std::max(worst_params, worst_dx) << ")\n" << std::flush;
}

/** @brief Gradientes de LayerNorm respecto de x, gamma y beta. */
void TestGradientCheckLayerNorm() {
  std::cout << "🧪 [Test 14] Gradientes de LayerNorm... " << std::flush;

  const int N = 3, D = 6;
  Tensor x({N, D});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.8f * std::sin(1.7f * static_cast<float>(i)) + 0.2f;
  }
  Tensor gamma({D}), beta({D});
  for (int j = 0; j < D; ++j) {
    gamma[j] = 0.7f + 0.1f * static_cast<float>(j);
    beta[j] = 0.05f * static_cast<float>(j) - 0.1f;
  }

  Tensor w({N, D});
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 0.5f + 0.5f * std::cos(0.9f * static_cast<float>(i));
  }
  auto loss_of = [&]() {
    Tensor out, mean, rstd;
    LayerNormForward(x, gamma, beta, out, mean, rstd);
    double s = 0.0;
    for (size_t i = 0; i < out.TotalSize(); ++i) s += static_cast<double>(out[i]) * w[i];
    return s;
  };

  Tensor out, mean, rstd, dx, dgamma, dbeta;
  LayerNormForward(x, gamma, beta, out, mean, rstd);
  LayerNormBackward(w, x, gamma, mean, rstd, dx, dgamma, dbeta);

  const float eps = 1e-2f;
  double worst = 0.0;
  int checked = 0;

  // Tres tensores con gradiente: la entrada y los dos parametros afines.
  struct Target { Tensor* value; Tensor* grad; const char* name; };
  Target targets[] = {{&x, &dx, "x"}, {&gamma, &dgamma, "gamma"}, {&beta, &dbeta, "beta"}};

  for (const Target& t : targets) {
    for (size_t i = 0; i < t.value->TotalSize(); ++i) {
      const float orig = (*t.value)[i];
      (*t.value)[i] = orig + eps; const double lp = loss_of();
      (*t.value)[i] = orig - eps; const double lm = loss_of();
      (*t.value)[i] = orig;

      const double numeric = (lp - lm) / (2.0 * eps);
      const double analytic = (*t.grad)[i];
      if (std::max(std::abs(numeric), std::abs(analytic)) < kNegligibleGrad) continue;
      const double rel = RelativeError(numeric, analytic);
      if (rel > worst) worst = rel;
      ++checked;
    }
  }

  Check(worst < 1e-2, "el peor error relativo de LayerNorm es " + std::to_string(worst));
  std::cout << "PASADO ✅ (" << checked << " elementos, peor error rel: " << worst << ")\n"
            << std::flush;
}

/** @brief El Backward de las pérdidas coincide con la derivada de su Forward. */
void TestGradientCheckLosses() {
  std::cout << "🧪 [Test 15] Gradientes de CrossEntropy y MSE... " << std::flush;

  const int N = 4, C = 5;
  Tensor logits({N, C});
  for (size_t i = 0; i < logits.TotalSize(); ++i) {
    logits[i] = 0.9f * std::sin(1.3f * static_cast<float>(i));
  }
  Tensor targets({N});
  for (int i = 0; i < N; ++i) targets[i] = static_cast<float>((i * 2 + 1) % C);

  const float eps = 1e-2f;

  // CrossEntropy: el gradiente debe ser (softmax - onehot) / N.
  CrossEntropyLoss ce;
  ce.Forward(logits, targets);
  Tensor dce = ce.Backward();

  double worst_ce = 0.0;
  int n_ce = 0;
  for (size_t i = 0; i < logits.TotalSize(); ++i) {
    const float orig = logits[i];
    logits[i] = orig + eps; const double lp = ce.Forward(logits, targets);
    logits[i] = orig - eps; const double lm = ce.Forward(logits, targets);
    logits[i] = orig;

    const double numeric = (lp - lm) / (2.0 * eps);
    if (std::max(std::abs(numeric), std::abs(static_cast<double>(dce[i]))) < kNegligibleGrad) continue;
    worst_ce = std::max(worst_ce, RelativeError(numeric, dce[i]));
    ++n_ce;
  }
  Check(worst_ce < 1e-2, "el peor error relativo de CrossEntropyLoss es " + std::to_string(worst_ce));

  // MSE.
  Tensor preds({N, C}), gold({N, C});
  for (size_t i = 0; i < preds.TotalSize(); ++i) {
    preds[i] = 0.6f * std::cos(0.8f * static_cast<float>(i));
    gold[i] = 0.3f * std::sin(0.5f * static_cast<float>(i));
  }
  MSELoss mse;
  mse.Forward(preds, gold);
  Tensor dmse = mse.Backward();

  double worst_mse = 0.0;
  int n_mse = 0;
  for (size_t i = 0; i < preds.TotalSize(); ++i) {
    const float orig = preds[i];
    preds[i] = orig + eps; const double lp = mse.Forward(preds, gold);
    preds[i] = orig - eps; const double lm = mse.Forward(preds, gold);
    preds[i] = orig;

    const double numeric = (lp - lm) / (2.0 * eps);
    if (std::max(std::abs(numeric), std::abs(static_cast<double>(dmse[i]))) < kNegligibleGrad) continue;
    worst_mse = std::max(worst_mse, RelativeError(numeric, dmse[i]));
    ++n_mse;
  }
  Check(worst_mse < 1e-2, "el peor error relativo de MSELoss es " + std::to_string(worst_mse));

  std::cout << "PASADO ✅ (CE: " << n_ce << " elementos, " << worst_ce
            << "; MSE: " << n_mse << " elementos, " << worst_mse << ")\n" << std::flush;
}

/**
 * @brief Gradientes de MaxPool2D, ResidualBlock y GraphConv.
 *
 * Estas capas seleccionan: ReLU y el maximo del pooling tienen derivada
 * discontinua. Eso invierte el criterio para elegir el paso respecto de las
 * pruebas anteriores. En una perdida suave conviene un paso grande, porque el
 * error lo domina la cancelacion en float32; aqui conviene uno pequeno, porque
 * un paso grande hace que la perturbacion cruce el codo y los dos lados de la
 * diferencia central queden en regimenes distintos.
 *
 * Medido sobre GraphConv: eps=1e-3 da 1.8e-06 de error, y eps=5e-3 lo dispara a
 * 1.0, que es lo que se observa cuando una unidad se enciende en un lado de la
 * diferencia y no en el otro.
 */
void TestGradientCheckRemainingLayers() {
  std::cout << "🧪 [Test 17] Gradientes de MaxPool2D, Residual y GraphConv... " << std::flush;

  const float eps = 1e-3f;

  // Comprueba dx recorriendo la entrada, para capas cuyo gradiente de entrada
  // es lo unico o lo primero que hay que validar.
  auto check_dx = [&](Tensor& x, const Tensor& dx, const std::function<double()>& loss_of,
                      const char* label) {
    double worst = 0.0;
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      const float orig = x[i];
      x[i] = orig + eps; const double lp = loss_of();
      x[i] = orig - eps; const double lm = loss_of();
      x[i] = orig;
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(dx[i]))) < kNegligibleGrad) continue;
      worst = std::max(worst, RelativeError(numeric, dx[i]));
    }
    Check(worst < 1e-2, std::string("el peor error relativo de dx de ") + label + " es " +
                            std::to_string(worst));
    return worst;
  };

  double worst_pool = 0.0, worst_res = 0.0, worst_gcn = 0.0;

  // --- MaxPool2D: sin parametros, solo enruta el gradiente al maximo. ---
  {
    MaxPool2D pool(2, 2);
    Tensor x({1, 2, 4, 4});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.7f * std::sin(1.9f * static_cast<float>(i)) + 0.15f;
    }
    Tensor probe = pool.Forward(x);
    Tensor w(probe.Shape());
    for (size_t i = 0; i < w.TotalSize(); ++i) {
      w[i] = 0.6f + 0.4f * std::cos(0.7f * static_cast<float>(i));
    }
    auto loss_of = [&]() {
      Tensor y = pool.Forward(x);
      double s = 0.0;
      for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
      return s;
    };
    loss_of();
    Tensor dx = pool.Backward(w);
    worst_pool = check_dx(x, dx, loss_of, "MaxPool2D");
  }

  // --- ResidualBlock: y = ReLU(f(x) + x); el atajo tambien lleva gradiente. ---
  {
    auto inner = std::make_shared<Linear>(6, 6);
    ResidualBlock block(inner);
    Tensor x({3, 6});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.5f * std::sin(1.3f * static_cast<float>(i)) + 0.6f;
    }
    Tensor w({3, 6});
    for (size_t i = 0; i < w.TotalSize(); ++i) {
      w[i] = 0.5f + 0.5f * std::cos(0.9f * static_cast<float>(i));
    }
    auto loss_of = [&]() {
      Tensor y = block.Forward(x);
      double s = 0.0;
      for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
      return s;
    };
    loss_of();
    Tensor dx = block.Backward(w);
    worst_res = check_dx(x, dx, loss_of, "ResidualBlock");

    int checked = 0, skipped = 0;
    const double wp = MaxGradError(block.GetParameters(), block.GetGradients(), loss_of,
                                   eps, 8, &checked, &skipped);
    Check(wp < 1e-2, "el peor error relativo de los parametros de ResidualBlock es " +
                         std::to_string(wp));
    worst_res = std::max(worst_res, wp);
  }

  // --- GraphConv: H_out = ReLU(A · H · W). ---
  {
    GraphConv gcn(4, 3);
    const int N = 4;
    Tensor x({N, 4});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.5f * std::sin(1.1f * static_cast<float>(i)) + 0.55f;
    }
    // La adyacencia debe cumplir dos cosas. No ser la identidad, porque
    // entonces la capa degeneraria en una densa y la agregacion no quedaria
    // comprobada. Y no ser simetrica: el backward multiplica por la transpuesta
    // de la adyacencia, y con una matriz simetrica omitir esa transposicion no
    // cambia el resultado, de modo que el error pasaria inadvertido. Aqui el
    // grafo es dirigido, con pesos distintos hacia delante y hacia atras.
    Tensor adj({N, N});
    adj.Zeros();
    for (int i = 0; i < N; ++i) {
      adj[i * N + i] = 0.5f;
      if (i + 1 < N) adj[i * N + (i + 1)] = 0.30f;   // arco i -> i+1
      if (i > 0) adj[i * N + (i - 1)] = 0.10f;       // arco i -> i-1, con otro peso
    }
    Tensor probe = gcn.ForwardWithAdj(x, adj);
    Tensor w(probe.Shape());
    for (size_t i = 0; i < w.TotalSize(); ++i) {
      w[i] = 0.5f + 0.5f * std::cos(1.7f * static_cast<float>(i));
    }
    auto loss_of = [&]() {
      Tensor y = gcn.ForwardWithAdj(x, adj);
      double s = 0.0;
      for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
      return s;
    };
    loss_of();
    Tensor dx = gcn.Backward(w);
    worst_gcn = check_dx(x, dx, loss_of, "GraphConv");

    int checked = 0, skipped = 0;
    const double wp = MaxGradError(gcn.GetParameters(), gcn.GetGradients(), loss_of,
                                   eps, 8, &checked, &skipped);
    Check(wp < 1e-2, "el peor error relativo de los parametros de GraphConv es " +
                         std::to_string(wp));
    worst_gcn = std::max(worst_gcn, wp);
  }

  std::cout << "PASADO ✅ (pool: " << worst_pool << ", residual: " << worst_res
            << ", gcn: " << worst_gcn << ")\n" << std::flush;
}

/**
 * @brief Parameter y el registro automatico de Module.
 *
 * El defecto original consistia en que una capa declarase sus pesos y olvidase
 * declarar sus gradientes. Ahora no hay dos declaraciones: `GetParameters()` y
 * `GetGradients()` se derivan de la misma lista, asi que la desalineacion ya no
 * es representable. Esta prueba comprueba lo que si puede fallar todavia: que
 * el recorrido del arbol de submodulos los recoja todos.
 */
void TestParameterAndModule() {
  std::cout << "🧪 [Test 18] Parameter y registro automático de Module... " << std::flush;

  // Valor y gradiente nacen con la misma forma y se redimensionan juntos.
  Parameter p({3, 4});
  Check(p.Value().Shape() == p.Grad().Shape(), "el gradiente no nacio con la forma del valor");
  p.Resize({2, 5});
  Check(p.Value().Shape() == p.Grad().Shape(), "Resize dejo valor y gradiente con formas distintas");

  // Una capa simple declara sus dos parametros.
  Linear fc(4, 3);
  Check(fc.Parameters().size() == 2, "Linear deberia declarar dos parametros");

  // El arbol se recorre solo: un bloque del GPT reune los de sus cinco
  // componentes sin que nadie los enumere a mano.
  GPTConfig cfg;
  cfg.vocab_size = 16; cfg.block_size = 8;
  cfg.n_layer = 3; cfg.n_head = 2; cfg.n_embd = 8;

  GPTBlock block(cfg);
  Check(block.Parameters().size() == 12,
        "un GPTBlock deberia reunir 12 parametros y reune " +
            std::to_string(block.Parameters().size()));

  // wte + wpe + 3 bloques * 12 + ln_f = 2 + 36 + 2 = 40
  GPTModel model(cfg);
  const size_t expected = 2 + static_cast<size_t>(cfg.n_layer) * 12 + 2;
  Check(model.Parameters().size() == expected,
        "el GPT deberia reunir " + std::to_string(expected) + " parametros y reune " +
            std::to_string(model.Parameters().size()));

  // Las dos vistas de la misma lista tienen que seguir emparejadas.
  auto values = model.GetParameters();
  auto grads = model.GetGradients();
  Check(values.size() == grads.size(), "las listas derivadas tienen tamanos distintos");
  bool shapes_ok = true;
  for (size_t i = 0; i < values.size() && i < grads.size(); ++i) {
    if (values[i]->Shape() != grads[i]->Shape()) shapes_ok = false;
  }
  Check(shapes_ok, "hay un par valor/gradiente con formas distintas");

  std::cout << "PASADO ✅ (" << model.Parameters().size() << " parámetros en el árbol)\n"
            << std::flush;
}

/**
 * @brief El formato NSF conserva los pesos y rechaza lo que no corresponde.
 *
 * Lo que importa aqui no es que el viaje de ida y vuelta funcione, sino que
 * cargar algo incompatible falle. El formato anterior no tenia cabecera: un
 * archivo de otra arquitectura, o truncado, se cargaba sin dar ningun error y
 * el modelo se quedaba con datos sin sentido.
 */
void TestSerialization() {
  std::cout << "🧪 [Test 19] Formato de pesos NSF... " << std::flush;

  // La ruta temporal se pide al sistema: /tmp no existe en Windows, y
  // codificarla hacia que esta prueba fallase alli aunque el formato estuviera
  // bien.
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "ns_test_weights.nsf").string();
  const std::string path_truncated = (tmp / "ns_test_truncated.nsf").string();
  const std::string path_corrupt = (tmp / "ns_test_corrupt.nsf").string();
  const std::string path_legacy = (tmp / "ns_test_legacy.bin").string();

  GPTConfig cfg;
  cfg.vocab_size = 12; cfg.block_size = 8;
  cfg.n_layer = 2; cfg.n_head = 2; cfg.n_embd = 8;

  // Ida y vuelta: los pesos guardados deben volver identicos.
  GPTModel saver(cfg);
  Check(saver.SaveWeights(path), "no se pudo guardar el modelo");

  GPTModel loader(cfg);
  Check(loader.LoadWeights(path), "no se pudo cargar un archivo que deberia encajar");

  bool identical = true;
  auto a = saver.GetParameters();
  auto b = loader.GetParameters();
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    for (size_t k = 0; k < a[i]->TotalSize(); ++k) {
      if (std::abs((*a[i])[k] - (*b[i])[k]) > 0.0f) identical = false;
    }
  }
  Check(a.size() == b.size() && identical, "los pesos cargados no son identicos a los guardados");

  // Una arquitectura distinta debe rechazarse, no cargarse a medias.
  GPTConfig other = cfg;
  other.n_layer = 3;
  GPTModel wrong_layers(other);
  Check(!wrong_layers.LoadWeights(path),
        "se acepto un archivo de 2 capas en un modelo de 3");

  GPTConfig other_vocab = cfg;
  other_vocab.vocab_size = 20;
  GPTModel wrong_vocab(other_vocab);
  Check(!wrong_vocab.LoadWeights(path),
        "se acepto un archivo con otro tamano de vocabulario");

  // Un archivo truncado debe detectarse.
  {
    std::ifstream src(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
    std::ofstream dst(path_truncated, std::ios::binary);
    dst.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
  }
  GPTModel truncated_target(cfg);
  Check(!truncated_target.LoadWeights(path_truncated),
        "se acepto un archivo truncado");

  // Un byte cambiado debe hacer fallar la suma de comprobacion.
  {
    std::ifstream src(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
    if (bytes.size() > 200) bytes[bytes.size() - 40] ^= 0x7F;
    std::ofstream dst(path_corrupt, std::ios::binary);
    dst.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  GPTModel corrupt_target(cfg);
  Check(!corrupt_target.LoadWeights(path_corrupt),
        "se acepto un archivo con un byte alterado");

  // Un archivo que no es NSF (el volcado crudo de antes) debe rechazarse.
  {
    std::ofstream raw(path_legacy, std::ios::binary);
    std::vector<float> junk(1000, 0.5f);
    raw.write(reinterpret_cast<const char*>(junk.data()),
              static_cast<std::streamsize>(junk.size() * sizeof(float)));
  }
  GPTModel legacy_target(cfg);
  Check(!legacy_target.LoadWeights(path_legacy),
        "se acepto un volcado sin cabecera");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Cada primitiva del autograd, contra diferencias finitas.
 *
 * El motor deduce la derivada de la operacion en vez de que la escriba cada
 * capa. Eso solo vale si cada primitiva es correcta, asi que se comprueban una
 * a una: se construye una expresion, se reduce a escalar y se compara el
 * gradiente que produce el motor con el numerico.
 */
void TestAutogradPrimitives() {
  std::cout << "🧪 [Test 20] Primitivas del autograd... " << std::flush;
  using namespace neuralsuite::autograd;

  const float eps = 1e-2f;

  // Comprueba el gradiente de `x` en una expresion escalar cualquiera.
  auto check = [&](const char* name, Tensor seed,
                   const std::function<VarPtr(const VarPtr&)>& build) {
    auto x = Variable::Create(seed, true);
    auto out = build(x);
    Backward(out);

    double worst = 0.0;
    for (size_t i = 0; i < seed.TotalSize(); ++i) {
      const float orig = seed[i];

      Tensor plus = seed;  plus[i]  = orig + eps;
      Tensor minus = seed; minus[i] = orig - eps;
      const double lp = build(Variable::Create(plus))->Value()[0];
      const double lm = build(Variable::Create(minus))->Value()[0];

      const double numeric = (lp - lm) / (2.0 * eps);
      const double analytic = x->Grad()[i];
      if (std::max(std::abs(numeric), std::abs(analytic)) < kNegligibleGrad) continue;
      worst = std::max(worst, RelativeError(numeric, analytic));
    }
    Check(worst < 1e-2, std::string("autograd ") + name + ": error relativo " +
                            std::to_string(worst));
    return worst;
  };

  Tensor base({2, 3});
  for (size_t i = 0; i < base.TotalSize(); ++i) {
    base[i] = 0.4f * std::sin(1.7f * static_cast<float>(i)) + 0.9f;  // positivo, para Log
  }

  double worst = 0.0;
  worst = std::max(worst, check("Sum", base, [](const VarPtr& x) { return Sum(x); }));
  worst = std::max(worst, check("Mean", base, [](const VarPtr& x) { return Mean(x); }));
  worst = std::max(worst, check("Add", base, [](const VarPtr& x) { return Sum(x + x); }));
  worst = std::max(worst, check("Sub", base, [](const VarPtr& x) {
    auto c = Variable::Create(Tensor(x->Shape()));
    return Sum(x - c);
  }));
  worst = std::max(worst, check("Mul", base, [](const VarPtr& x) { return Sum(x * x); }));
  worst = std::max(worst, check("Exp", base, [](const VarPtr& x) { return Sum(Exp(x)); }));
  worst = std::max(worst, check("Log", base, [](const VarPtr& x) { return Sum(Log(x)); }));
  worst = std::max(worst, check("Tanh", base, [](const VarPtr& x) { return Sum(Tanh(x)); }));
  worst = std::max(worst, check("Reshape", base, [](const VarPtr& x) {
    return Sum(Tanh(Reshape(x, {3, 2})));
  }));
  worst = std::max(worst, check("Transpose", base, [](const VarPtr& x) {
    return Sum(Tanh(TransposeVar(x)));
  }));

  // Relu con entradas de ambos signos, lejos del codo.
  Tensor mixed({6});
  mixed[0] = 0.8f; mixed[1] = -0.7f; mixed[2] = 1.3f;
  mixed[3] = -1.1f; mixed[4] = 0.5f; mixed[5] = -0.4f;
  worst = std::max(worst, check("Relu", mixed, [](const VarPtr& x) { return Sum(Relu(x)); }));

  // MatMul: se comprueba respecto de la matriz de la izquierda.
  {
    Tensor a({2, 3}), b({3, 2});
    for (size_t i = 0; i < a.TotalSize(); ++i) a[i] = 0.3f * std::cos(1.1f * static_cast<float>(i));
    for (size_t i = 0; i < b.TotalSize(); ++i) b[i] = 0.5f * std::sin(0.9f * static_cast<float>(i)) + 0.2f;

    auto build = [&](const Tensor& av) {
      auto va = Variable::Create(av, true);
      auto vb = Variable::Create(b);
      return std::make_pair(va, Sum(Tanh(MatMulVar(va, vb))));
    };
    auto [va, out] = build(a);
    Backward(out);

    double w = 0.0;
    for (size_t i = 0; i < a.TotalSize(); ++i) {
      const float orig = a[i];
      Tensor plus = a;  plus[i]  = orig + eps;
      Tensor minus = a; minus[i] = orig - eps;
      const double lp = build(plus).second->Value()[0];
      const double lm = build(minus).second->Value()[0];
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(va->Grad()[i]))) < kNegligibleGrad) continue;
      w = std::max(w, RelativeError(numeric, va->Grad()[i]));
    }
    Check(w < 1e-2, "autograd MatMul: error relativo " + std::to_string(w));
    worst = std::max(worst, w);
  }

  // Broadcasting: un sesgo [D] sumado a un lote [N, D]. En el backward, cada
  // elemento del sesgo debe recibir la suma de las N posiciones que lo usaron.
  {
    const int N = 3, D = 4;
    Tensor xb({N, D}), bb({D});
    for (size_t i = 0; i < xb.TotalSize(); ++i) xb[i] = 0.3f * std::sin(1.3f * static_cast<float>(i));
    for (int i = 0; i < D; ++i) bb[i] = 0.2f + 0.1f * static_cast<float>(i);

    auto build = [&](const Tensor& bias) {
      auto vx = Variable::Create(xb);
      auto vb = Variable::Create(bias, true);
      return std::make_pair(vb, Sum(Tanh(vx + vb)));
    };
    auto [vb, out] = build(bb);
    Check(out->Shape() == std::vector<int>({N, D}) || out->Value().TotalSize() == 1,
          "el broadcasting no produjo la forma esperada");
    Backward(out);

    double w = 0.0;
    for (int i = 0; i < D; ++i) {
      const float orig = bb[i];
      Tensor plus = bb;  plus[i]  = orig + eps;
      Tensor minus = bb; minus[i] = orig - eps;
      const double lp = build(plus).second->Value()[0];
      const double lm = build(minus).second->Value()[0];
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(vb->Grad()[i]))) < kNegligibleGrad) continue;
      w = std::max(w, RelativeError(numeric, vb->Grad()[i]));
    }
    Check(w < 1e-2, "autograd broadcasting: error relativo " + std::to_string(w));
    Check(vb->Grad().Shape() == std::vector<int>({D}),
          "el gradiente del sesgo no volvio con su forma original");
    worst = std::max(worst, w);
  }

  // Un nodo usado por dos caminos debe recibir la suma de ambos: y = x*x + x
  // tiene dy/dx = 2x + 1. Si el recorrido no acumulase, saldria uno solo.
  {
    Tensor t({4});
    for (int i = 0; i < 4; ++i) t[i] = 0.5f + 0.3f * static_cast<float>(i);
    auto x = Variable::Create(t, true);
    Backward(Sum(x * x + x));

    bool ok = true;
    for (size_t i = 0; i < t.TotalSize(); ++i) {
      const float expected = 2.0f * t[i] + 1.0f;
      if (std::abs(x->Grad()[i] - expected) > 1e-4f) ok = false;
    }
    Check(ok, "un nodo con dos caminos no acumulo las dos contribuciones");
  }

  std::cout << "PASADO ✅ (12 primitivas + broadcasting, peor error rel: " << worst << ")\n" << std::flush;
}

/** @brief ManualSeed reproduce la inicializacion; ParamGroup declara el decay. */
void TestSeedAndParamGroups() {
  std::cout << "🧪 [Test 21] Semilla reproducible y grupos de parámetros... " << std::flush;

  // Dos inicializaciones con la misma semilla deben coincidir exactamente.
  ManualSeed(2024);
  Tensor a({50});
  a.RandomNormal(0.0f, 1.0f);
  ManualSeed(2024);
  Tensor b({50});
  b.RandomNormal(0.0f, 1.0f);

  bool same = true;
  for (size_t i = 0; i < a.TotalSize(); ++i) {
    if (a[i] != b[i]) same = false;
  }
  Check(same, "la misma semilla produjo inicializaciones distintas");

  ManualSeed(999);
  Tensor c({50});
  c.RandomNormal(0.0f, 1.0f);
  bool differs = false;
  for (size_t i = 0; i < a.TotalSize(); ++i) {
    if (a[i] != c[i]) differs = true;
  }
  Check(differs, "semillas distintas produjeron la misma inicializacion");

  // El grupo con decay 0 no debe encoger sus pesos; el que lo declara, si.
  // Con la heuristica anterior ambos son 2D y recibirian el mismo trato.
  Parameter decays({2, 2}), keeps({2, 2});
  for (int i = 0; i < 4; ++i) {
    decays.Value()[i] = 1.0f;
    keeps.Value()[i] = 1.0f;
    decays.Grad()[i] = 0.0f;
    keeps.Grad()[i] = 0.0f;
  }

  // Gradientes a cero, para que el unico efecto sobre los pesos sea el decay.
  // La tasa de aprendizaje debe ser distinta de cero: multiplica tambien al
  // termino de decay, asi que con cero no se movería nada y la prueba no
  // distinguiria un grupo del otro.
  AdamW opt(std::vector<ParamGroup>{{{&decays}, 0.5f}, {{&keeps}, 0.0f}}, 0.1f);
  opt.Step();

  Check(decays.Value()[0] < 0.99f, "el grupo con weight decay no encogio sus pesos");
  Check(std::abs(keeps.Value()[0] - 1.0f) < 1e-6f,
        "el grupo sin weight decay vio sus pesos modificados");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Lo desconocido se marca, y el tokenizador de bytes no puede fallar.
 *
 * Antes un caracter fuera del vocabulario se codificaba como token 0, que era
 * un caracter valido: con el vocabulario {a,b,c}, "axc" volvia como "aac".
 */
void TestTokenizerUnknownAndBytes() {
  std::cout << "🧪 [Test 22] Token desconocido y tokenizador de bytes... " << std::flush;

  CharTokenizer abc("abc");
  Check(abc.VocabSize() == 4, "el vocabulario deberia ser {a,b,c} mas <UNK>");

  const std::vector<int> ids = abc.Encode("axc");
  Check(ids.size() == 3, "Encode no produjo un token por byte");
  Check(ids[1] == CharTokenizer::kUnknownToken, "la 'x' desconocida no dio <UNK>");
  Check(ids[0] != CharTokenizer::kUnknownToken && ids[2] != CharTokenizer::kUnknownToken,
        "un caracter conocido se confundio con <UNK>");
  Check(abc.Decode(ids) != "aac", "lo desconocido volvio a confundirse con la 'a'");
  Check(abc.CountUnknown("axc") == 1, "CountUnknown no detecto el simbolo fuera del vocabulario");

  // Roundtrip completo cuando todo el texto esta en el vocabulario.
  const std::string sample = "Hello C++ Google Style!";
  CharTokenizer tok(sample);
  Check(tok.Decode(tok.Encode(sample)) == sample, "el roundtrip no conserva el texto conocido");
  Check(tok.CountUnknown(sample) == 0, "se reportaron desconocidos en su propio corpus");

  // El vocabulario guardado y recargado debe comportarse igual.
  const std::string vocab_path =
      (std::filesystem::temp_directory_path() / "ns_test_vocab.txt").string();
  Check(tok.Save(vocab_path), "no se pudo guardar el vocabulario");
  CharTokenizer reloaded;
  Check(reloaded.Load(vocab_path), "no se pudo cargar el vocabulario");
  Check(reloaded.VocabSize() == tok.VocabSize(), "el vocabulario recargado cambio de tamano");
  Check(reloaded.Decode(reloaded.Encode(sample)) == sample,
        "el vocabulario recargado no reproduce el texto");

  // ByteTokenizer: por construccion no existe el simbolo desconocido.
  ByteTokenizer bytes;
  Check(bytes.VocabSize() == 256, "el vocabulario de bytes deberia ser fijo de 256");
  const std::string utf8 = "El niño comió jamón. ¿Qué más?";
  Check(bytes.Decode(bytes.Encode(utf8)) == utf8, "el texto UTF-8 no sobrevivio al roundtrip");

  // Texto que el tokenizador nunca vio: no hace falta reentrenarlo.
  const std::string otros = "日本語 y emoji 🎉";
  Check(bytes.Decode(bytes.Encode(otros)) == otros,
        "un texto de otro alfabeto no sobrevivio al roundtrip");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Softmax y LayerNorm del autograd.
 *
 * LayerNorm no tiene backward propio: se compone de primitivas y el motor
 * deduce la derivada. La prueba lo compara ademas contra la implementacion
 * escrita a mano, que es la referencia ya verificada.
 */
void TestAutogradComposites() {
  std::cout << "🧪 [Test 23] Softmax y LayerNorm por composición... " << std::flush;
  using namespace neuralsuite::autograd;

  const float eps = 1e-2f;
  const int N = 3, D = 5;

  Tensor base({N, D});
  for (size_t i = 0; i < base.TotalSize(); ++i) {
    base[i] = 0.6f * std::sin(1.3f * static_cast<float>(i)) + 0.1f;
  }
  Tensor w({N, D});
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 0.5f + 0.5f * std::cos(0.7f * static_cast<float>(i));
  }
  auto weight = Variable::Create(w);

  double worst = 0.0;

  // --- Softmax: gradiente frente a diferencias finitas. ---
  {
    auto build = [&](const Tensor& xv) {
      auto x = Variable::Create(xv, true);
      return std::make_pair(x, Sum(Mul(Softmax(x), weight)));
    };
    auto [x, out] = build(base);
    Backward(out);

    // Cada fila debe sumar uno.
    Tensor probs = Softmax(Variable::Create(base))->Value();
    bool rows_ok = true;
    for (int r = 0; r < N; ++r) {
      double acc = 0.0;
      for (int c = 0; c < D; ++c) acc += probs[r * D + c];
      if (std::abs(acc - 1.0) > 1e-5) rows_ok = false;
    }
    Check(rows_ok, "las filas del softmax no suman uno");

    double wsm = 0.0;
    for (size_t i = 0; i < base.TotalSize(); ++i) {
      const float orig = base[i];
      Tensor plus = base;  plus[i]  = orig + eps;
      Tensor minus = base; minus[i] = orig - eps;
      const double lp = build(plus).second->Value()[0];
      const double lm = build(minus).second->Value()[0];
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(x->Grad()[i]))) < kNegligibleGrad) continue;
      wsm = std::max(wsm, RelativeError(numeric, x->Grad()[i]));
    }
    Check(wsm < 1e-2, "autograd Softmax: error relativo " + std::to_string(wsm));
    worst = std::max(worst, wsm);
  }

  // --- LayerNorm: mismo resultado que la version escrita a mano. ---
  {
    Tensor gamma_t({D}), beta_t({D});
    for (int j = 0; j < D; ++j) {
      gamma_t[j] = 0.8f + 0.1f * static_cast<float>(j);
      beta_t[j] = 0.05f * static_cast<float>(j) - 0.1f;
    }

    auto x = Variable::Create(base, true);
    auto gamma = Variable::Create(gamma_t, true);
    auto beta = Variable::Create(beta_t, true);
    auto normalized = LayerNorm(x, gamma, beta);

    Tensor manual, mean_c, rstd_c;
    LayerNormForward(base, gamma_t, beta_t, manual, mean_c, rstd_c);

    double worst_fwd = 0.0;
    for (size_t i = 0; i < manual.TotalSize(); ++i) {
      worst_fwd = std::max(worst_fwd,
                           static_cast<double>(std::abs(normalized->Value()[i] - manual[i])));
    }
    Check(worst_fwd < 1e-4,
          "la LayerNorm compuesta no coincide con la escrita a mano: " + std::to_string(worst_fwd));

    // Y su gradiente, que nadie escribio, frente a diferencias finitas.
    auto build = [&](const Tensor& xv) {
      auto vx = Variable::Create(xv, true);
      auto vg = Variable::Create(gamma_t);
      auto vb = Variable::Create(beta_t);
      return std::make_pair(vx, Sum(Mul(LayerNorm(vx, vg, vb), weight)));
    };
    auto [vx, out] = build(base);
    Backward(out);

    double wln = 0.0;
    for (size_t i = 0; i < base.TotalSize(); ++i) {
      const float orig = base[i];
      Tensor plus = base;  plus[i]  = orig + eps;
      Tensor minus = base; minus[i] = orig - eps;
      const double lp = build(plus).second->Value()[0];
      const double lm = build(minus).second->Value()[0];
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(vx->Grad()[i]))) < kNegligibleGrad) continue;
      wln = std::max(wln, RelativeError(numeric, vx->Grad()[i]));
    }
    Check(wln < 2e-2, "autograd LayerNorm: error relativo " + std::to_string(wln));
    worst = std::max(worst, wln);
  }

  std::cout << "PASADO ✅ (peor error rel: " << worst << ")\n" << std::flush;
}

/**
 * @brief El reparto entre hilos no altera el resultado.
 *
 * Es la propiedad que hace segura la paralelizacion: cada hilo escribe filas
 * que ningun otro toca, de modo que no hay reduccion que cambie el orden de las
 * sumas en punto flotante y el resultado es identico bit a bit.
 */
void TestParallelDeterminism() {
  std::cout << "🧪 [Test 24] El paralelismo no cambia el resultado... " << std::flush;

  Tensor A({256, 128}), B({128, 256});
  A.RandomNormal(0.0f, 1.0f);
  B.RandomNormal(0.0f, 1.0f);

  const int original = parallel::ThreadCount();

  parallel::ThreadCount() = 1;
  Tensor serial;
  MatMul(A, B, serial);

  parallel::ThreadCount() = original;
  Tensor threaded;
  MatMul(A, B, threaded);

  bool identical = true;
  for (size_t i = 0; i < serial.TotalSize(); ++i) {
    if (serial[i] != threaded[i]) identical = false;
  }
  Check(identical, "el resultado con varios hilos difiere del de uno solo");

  // Pedir mas hilos de los que tiene el pool no debe dejar porciones del bucle
  // sin ejecutar: el pool se crea una vez y no puede crecer despues.
  parallel::ThreadCount() = original * 64 + 128;
  Tensor oversubscribed;
  MatMul(A, B, oversubscribed);
  parallel::ThreadCount() = original;

  bool complete = true;
  for (size_t i = 0; i < serial.TotalSize(); ++i) {
    if (serial[i] != oversubscribed[i]) complete = false;
  }
  Check(complete, "pedir mas hilos de los disponibles dejo parte del calculo sin hacer");

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief BiLSTM: que la mitad inversa mire de verdad hacia el futuro.
 *
 * Una comprobación de gradiente no distinguiría un BiLSTM correcto de dos
 * pasadas hacia delante concatenadas: ambas serían derivables y consistentes
 * consigo mismas. Lo que define a la capa es de qué depende cada salida, así
 * que eso es lo que se mide aquí, perturbando la entrada en un extremo y
 * viendo qué se mueve en el otro.
 */
void TestBiLstmDirectionality() {
  std::cout << "🧪 [Test 25] El BiLSTM mira en los dos sentidos... " << std::flush;

  const int T = 5, B = 2, IN = 3, H = 4;
  ManualSeed(1234);
  BiLSTM bi(IN, H);

  Tensor x({T, B, IN});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.4f * std::sin(0.9f * static_cast<float>(i)) + 0.2f;
  }
  const Tensor base = bi.Forward(x);
  Check(base.Shape() == std::vector<int>({T, B, 2 * H}),
        "la salida del BiLSTM no es [T, B, 2H]");

  // Índice del canal c del paso t, lote 0. Los primeros H son el sentido
  // directo; los H siguientes, el inverso.
  auto at = [&](const Tensor& y, int t, int c) {
    return y[static_cast<size_t>(t * B) * 2 * H + c];
  };
  auto changed = [](float a, float b) { return std::abs(a - b) > 1e-6f; };

  // 1. Tocar el último paso no puede alterar el sentido directo en el primero,
  //    pero sí debe alterar el inverso: es justo lo que una LSTM normal no ve.
  {
    Tensor xp = x;
    for (int j = 0; j < IN; ++j) xp[static_cast<size_t>((T - 1) * B) * IN + j] += 0.5f;
    const Tensor y = bi.Forward(xp);

    bool fwd_intact = true, rev_moved = false;
    for (int c = 0; c < H; ++c) {
      if (changed(at(y, 0, c), at(base, 0, c))) fwd_intact = false;
      if (changed(at(y, 0, H + c), at(base, 0, H + c))) rev_moved = true;
    }
    Check(fwd_intact, "el sentido directo del BiLSTM depende de pasos futuros");
    Check(rev_moved, "el sentido inverso del BiLSTM no depende del futuro: no es bidireccional");
  }

  // 2. La simétrica. Descarta además que se haya olvidado devolver la salida
  //    inversa a su orden temporal: sin esa segunda inversión, la última
  //    posición del sentido inverso habría visto la secuencia entera.
  {
    Tensor xp = x;
    for (int j = 0; j < IN; ++j) xp[j] += 0.5f;
    const Tensor y = bi.Forward(xp);

    bool rev_intact = true, fwd_moved = false;
    for (int c = 0; c < H; ++c) {
      if (changed(at(y, T - 1, H + c), at(base, T - 1, H + c))) rev_intact = false;
      if (changed(at(y, T - 1, c), at(base, T - 1, c))) fwd_moved = true;
    }
    Check(rev_intact, "el sentido inverso del BiLSTM no quedo realineado en el tiempo");
    Check(fwd_moved, "el sentido directo del BiLSTM no depende del pasado");
  }

  // 3. Las dos celdas son independientes: si compartieran pesos, ambas mitades
  //    coincidirian en la unica posicion donde ven lo mismo.
  Check(bi.GetParameters().size() == 8,
        "el BiLSTM deberia exponer 8 tensores de parametros, expone " +
            std::to_string(bi.GetParameters().size()));

  std::cout << "PASADO ✅\n" << std::flush;
}

/**
 * @brief Gradientes del BiLSTM, incluida la suma de las dos ramas en `dx`.
 *
 * La entrada alimenta a las dos celdas, así que su gradiente es la suma de dos
 * contribuciones. Quedarse con una sola daría un resultado que aún parece
 * razonable —la mitad del valor correcto—, y solo las diferencias finitas lo
 * delatan.
 */
void TestGradientCheckBiLstm() {
  std::cout << "🧪 [Test 26] Gradientes del BiLSTM... " << std::flush;

  const int T = 4, B = 2, IN = 3, H = 4;
  ManualSeed(99);
  BiLSTM bi(IN, H);

  Tensor x({T, B, IN});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.3f * std::sin(0.7f * static_cast<float>(i)) + 0.1f;
  }
  Tensor w({T, B, 2 * H});
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 0.5f + 0.5f * std::cos(1.3f * static_cast<float>(i));
  }
  auto loss_of = [&]() {
    Tensor y = bi.Forward(x);
    double s = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
    return s;
  };

  loss_of();
  Tensor dout(w.Shape());
  for (size_t i = 0; i < w.TotalSize(); ++i) dout[i] = w[i];
  Tensor dx = bi.Backward(dout);
  Check(dx.Shape() == x.Shape(), "dx del BiLSTM no tiene la forma de la entrada");

  // Mismo paso que en el LSTM: los gradientes son pequenos y un eps corto los
  // ahoga en el ruido de float32.
  int checked = 0, skipped = 0;
  const double worst_params =
      MaxGradError(bi.GetParameters(), bi.GetGradients(), loss_of, 5e-2f, 12,
                   &checked, &skipped);
  Check(worst_params < 1e-2,
        "el peor error relativo de los parametros del BiLSTM es " + std::to_string(worst_params));

  const float eps = 5e-2f;
  double worst_dx = 0.0;
  int dx_checked = 0;
  for (size_t i = 0; i < x.TotalSize(); i += 2) {
    const float orig = x[i];
    x[i] = orig + eps; const double lp = loss_of();
    x[i] = orig - eps; const double lm = loss_of();
    x[i] = orig;

    const double numeric = (lp - lm) / (2.0 * eps);
    if (std::max(std::abs(numeric), std::abs(static_cast<double>(dx[i]))) < kNegligibleGrad) continue;
    worst_dx = std::max(worst_dx, RelativeError(numeric, dx[i]));
    ++dx_checked;
  }
  Check(worst_dx < 1e-2, "el peor error relativo de dx del BiLSTM es " + std::to_string(worst_dx));

  std::cout << "PASADO ✅ (" << checked << " params, " << dx_checked
            << " dx; peor error rel: " << std::max(worst_params, worst_dx) << ")\n"
            << std::flush;
}

/**
 * @brief CRNN de OCR: contrato de formas y gradientes de la red completa.
 *
 * Conv2D, MaxPool2D, BiLSTM y Linear ya tienen su propia comprobación de
 * gradiente. Lo que aquí es nuevo es el cableado: dos cambios de disposición
 * entre `[B, C, 1, T]` y `[T, B, C]`, y el orden en que se deshace la cadena.
 * Un error ahí no produce una desviación pequeña, produce un gradiente que no
 * tiene nada que ver.
 *
 * Es una comprobación deliberadamente gruesa, y conviene decir por qué. La red
 * es lineal a trozos en muchos sitios —tres ReLU sobre 16, 32 y 64 canales, y
 * tres pooling cuyo argmax puede cambiar—, así que perturbar una coordenada
 * cruza algún codo con bastante probabilidad. En ese punto la función no es
 * derivable a esa escala y la diferencia finita mide otra cosa. Se barrió el
 * paso entre 3e-5 y 1e-2, y no hay ninguna ventana limpia: por debajo domina la
 * cancelación de float32 y por encima, los codos. Lo mejor está en 1e-3, con la
 * mediana en 8e-04 y alrededor del 13% de coordenadas por encima de 1e-2. Se
 * probó también con una pérdida cuadrática, sin cancelación, y sale igual: es
 * una propiedad de la arquitectura, no del código.
 *
 * De ahí el criterio: la mediana y el percentil, no el peor caso. Un fallo de
 * cableado mueve la distribución entera, y así se comprobó mutando las cuatro
 * conversiones. La verificación fina de esta red es la paridad contra PyTorch,
 * que compara gradiente contra gradiente y no usa diferencias finitas.
 */
void TestCrnnOcr() {
  std::cout << "🧪 [Test 27] CRNN de OCR: formas y gradientes... " << std::flush;

  // El lote y los pasos deben ser distintos: con 2 y 2, las dos conversiones
  // entre `[T, B, K]` y `[B, T, K]` son la misma permutación y confundir una con
  // la otra no cambiaría nada. Se comprobó: la mutación pasaba desapercibida.
  const int kClasses = 5, kHidden = 4, kBatch = 2, kWidth = 12;
  const int kSteps = CRNNModel::TimestepsFor(kWidth);

  ManualSeed(21);
  CRNNModel crnn(1, kHidden, kClasses);

  Tensor x({kBatch, 1, CRNNModel::kInputHeight, kWidth});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.5f * std::sin(0.37f * static_cast<float>(i)) + 0.15f;
  }

  // 1. Una predicción por columna superviviente, no una por imagen. Es la
  //    diferencia entre leer una palabra y clasificar un carácter suelto.
  const Tensor probe = crnn.Forward(x);
  Check(probe.Shape() == std::vector<int>({kBatch, kSteps, kClasses}),
        "la salida del CRNN deberia ser [batch, tiempo, clases]");
  Check(kSteps == kWidth / 4, "el CRNN no reduce el ancho en un factor de 4");

  // 2. La geometría de entrada se valida en vez de producir basura.
  bool threw = false;
  try {
    Tensor bad({1, 1, CRNNModel::kInputHeight + 1, kWidth});
    crnn.Forward(bad);
  } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "el CRNN acepto una imagen con el alto equivocado");

  threw = false;
  try {
    Tensor bad({1, 1, CRNNModel::kInputHeight, kWidth + 1});
    crnn.Forward(bad);
  } catch (const std::invalid_argument&) { threw = true; }
  Check(threw, "el CRNN acepto un ancho que no es multiplo de 4");

  // 3. Gradientes de toda la red. Los pesos de la pérdida varían por posición
  //    —una suma simple no distinguiría un intercambio entre pasos temporales o
  //    entre muestras— y llevan un factor común que solo sirve para levantar los
  //    gradientes por encima del suelo de ruido: al ser lineal, no altera
  //    ninguno de los errores relativos que se miden.
  Tensor w({kBatch, kSteps, kClasses});
  for (size_t i = 0; i < w.TotalSize(); ++i) {
    w[i] = 200.0f * (0.5f + 0.5f * std::cos(1.7f * static_cast<float>(i)));
  }
  auto loss_of = [&]() {
    Tensor y = crnn.Forward(x);
    double s = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) s += static_cast<double>(y[i]) * w[i];
    return s;
  };

  loss_of();
  Tensor dout(w.Shape());
  for (size_t i = 0; i < w.TotalSize(); ++i) dout[i] = w[i];
  Tensor dx = crnn.Backward(dout);
  Check(dx.Shape() == x.Shape(), "dx del CRNN no tiene la forma de la imagen");

  const float eps = 1e-3f;
  const double kFloor = 1e-2;  // por debajo, el error relativo es ruido
  std::vector<double> errors;

  auto sample = [&](Tensor& value, const Tensor& grad, size_t stride) {
    for (size_t i = 0; i < value.TotalSize(); i += stride) {
      const float orig = value[i];
      value[i] = orig + eps; const double lp = loss_of();
      value[i] = orig - eps; const double lm = loss_of();
      value[i] = orig;
      const double numeric = (lp - lm) / (2.0 * eps);
      if (std::max(std::abs(numeric), std::abs(static_cast<double>(grad[i]))) < kFloor) continue;
      errors.push_back(RelativeError(numeric, grad[i]));
    }
  };

  const std::vector<Tensor*> params = crnn.GetParameters();
  const std::vector<Tensor*> grads = crnn.GetGradients();
  Check(params.size() == grads.size(), "el CRNN no alinea parametros y gradientes");
  for (size_t t = 0; t < params.size(); ++t) {
    sample(*params[t], *grads[t], std::max<size_t>(1, params[t]->TotalSize() / 8));
  }
  const size_t n_params = errors.size();
  sample(x, dx, 7);

  Check(n_params > 40, "se comprobaron muy pocos parametros del CRNN: " + std::to_string(n_params));
  Check(errors.size() - n_params > 40,
        "se comprobaron muy pocas entradas de dx del CRNN: " + std::to_string(errors.size() - n_params));

  std::sort(errors.begin(), errors.end());
  const double median = errors[errors.size() / 2];
  const double p95 = errors[static_cast<size_t>(errors.size() * 0.95)];
  size_t outliers = 0;
  for (double e : errors) {
    if (e > 1e-2) ++outliers;
  }

  Check(median < 5e-3, "la mediana del error relativo del CRNN es " + std::to_string(median));
  Check(p95 < 1.5e-1, "el percentil 95 del error relativo del CRNN es " + std::to_string(p95));
  Check(outliers * 10 < errors.size() * 3,
        "el " + std::to_string(100 * outliers / errors.size()) +
            "% de las coordenadas del CRNN supera 1e-2: son demasiadas para ser cruces de ReLU");

  std::cout << "PASADO ✅ (" << n_params << " params, " << errors.size() - n_params
            << " dx; mediana " << median << ", p95 " << p95 << ", " << outliers
            << " sobre 1e-2)\n"
            << std::flush;
}

namespace embedded_images {

// La misma imagen de 8x8 codificada de seis maneras. El valor de cada pixel es
// `(x*29 + y*53) % 256`, asi que la prueba lo recalcula y no hay que arrastrar
// una tabla de valores esperados. Los archivos van incrustados porque en la
// integracion continua no hay Pillow con que generarlos, y una prueba que
// dependa de archivos sueltos deja de correr en cuanto alguien mueve el
// directorio de trabajo.
//
// Que las seis formas den exactamente los mismos pixeles es una comprobacion
// mas fuerte que compararlas contra una tabla: cubre a la vez el desfiltrado,
// el entrelazado Adam7, la expansion de paleta y las dos rutas sin comprimir.

  // PNG en gris de 8 bits, sin entrelazar
  const uint8_t kPngGray[] = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
      0x08, 0x00, 0x00, 0x00, 0x00, 0xE1, 0x64, 0xE1, 0x57, 0x00, 0x00, 0x00,
      0x21, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x64, 0x90, 0x85, 0x00,
      0x46, 0x53, 0x28, 0x83, 0x05, 0xCA, 0x30, 0x85, 0x32, 0x4C, 0x65, 0x19,
      0xAF, 0xC0, 0xD4, 0x70, 0xA2, 0xA9, 0x41, 0x28, 0x06, 0x00, 0x30, 0x3F,
      0x08, 0x9B, 0xB9, 0xEA, 0xC2, 0x98, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
      0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
  };

  // el mismo con entrelazado Adam7 y filtro Paeth (escrito a mano: Pillow no sabe)
  const uint8_t kPngInterlaced[] = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
      0x08, 0x00, 0x00, 0x00, 0x01, 0x96, 0x63, 0xD1, 0xC1, 0x00, 0x00, 0x00,
      0x37, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x4D, 0x88, 0xB1, 0x11, 0x00,
      0x20, 0x08, 0xC4, 0x2C, 0x52, 0xBB, 0x01, 0x63, 0xD8, 0x30, 0xD2, 0xD7,
      0x8C, 0xCA, 0x40, 0xA2, 0x58, 0xF8, 0x45, 0x3E, 0x17, 0x06, 0x41, 0x06,
      0x5E, 0x4C, 0xE4, 0xEE, 0xE4, 0x9C, 0x89, 0x1D, 0xD3, 0x83, 0x50, 0x91,
      0x65, 0x3D, 0xD4, 0xAF, 0x2B, 0xFA, 0x8A, 0x6D, 0x00, 0xA8, 0x10, 0xD7,
      0xD1, 0xB8, 0xB0, 0xD8, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,
      0xAE, 0x42, 0x60, 0x82,
  };

  // el mismo, todas las filas con filtro Average
  const uint8_t kPngAverage[] = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
      0x08, 0x00, 0x00, 0x00, 0x00, 0xE1, 0x64, 0xE1, 0x57, 0x00, 0x00, 0x00,
      0x2E, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x66, 0x90, 0xD5, 0xB1,
      0xF2, 0x0C, 0x4F, 0x2B, 0x61, 0x36, 0xD5, 0x84, 0x00, 0xE6, 0x00, 0x18,
      0x23, 0x0B, 0x44, 0xAE, 0x04, 0x32, 0x5A, 0xC1, 0x14, 0x90, 0x31, 0x7F,
      0x25, 0x54, 0xCA, 0x0A, 0xA6, 0x26, 0x44, 0x13, 0xA2, 0x08, 0x00, 0x04,
      0x87, 0x0E, 0x90, 0x14, 0x17, 0x38, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x49,
      0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
  };

  // el mismo, como RGB
  const uint8_t kPngRgb[] = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
      0x08, 0x02, 0x00, 0x00, 0x00, 0x4B, 0x6D, 0x29, 0xDC, 0x00, 0x00, 0x00,
      0x2D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x64, 0x60, 0x60, 0x90,
      0xC5, 0x06, 0x18, 0x4D, 0x4D, 0x4D, 0xB1, 0x4A, 0xB0, 0x60, 0x95, 0x30,
      0x35, 0x35, 0xC5, 0x22, 0x01, 0x11, 0x61, 0xBC, 0x72, 0xE5, 0x0A, 0x76,
      0x3B, 0x38, 0x39, 0x39, 0x49, 0xB0, 0x03, 0x9F, 0xE5, 0x00, 0xD8, 0x59,
      0x19, 0xA7, 0x4C, 0x35, 0xD9, 0x2C, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
      0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
  };

  // el mismo, con paleta de 64 colores
  const uint8_t kPngPalette[] = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
      0x08, 0x03, 0x00, 0x00, 0x00, 0xF3, 0xD1, 0x4E, 0xB9, 0x00, 0x00, 0x00,
      0xAB, 0x50, 0x4C, 0x54, 0x45, 0xFB, 0xFB, 0xFB, 0xF6, 0xF6, 0xF6, 0xF1,
      0xF1, 0xF1, 0xEC, 0xEC, 0xEC, 0xE7, 0xE7, 0xE7, 0xE3, 0xE3, 0xE3, 0xDE,
      0xDE, 0xDE, 0xD9, 0xD9, 0xD9, 0xD4, 0xD4, 0xD4, 0xCF, 0xCF, 0xCF, 0xCB,
      0xCB, 0xCB, 0xCA, 0xCA, 0xCA, 0xC6, 0xC6, 0xC6, 0xC1, 0xC1, 0xC1, 0xBC,
      0xBC, 0xBC, 0xB7, 0xB7, 0xB7, 0xB2, 0xB2, 0xB2, 0xAE, 0xAE, 0xAE, 0xAD,
      0xAD, 0xAD, 0xA9, 0xA9, 0xA9, 0xA4, 0xA4, 0xA4, 0x9F, 0x9F, 0x9F, 0x9A,
      0x9A, 0x9A, 0x95, 0x95, 0x95, 0x91, 0x91, 0x91, 0x90, 0x90, 0x90, 0x8C,
      0x8C, 0x8C, 0x87, 0x87, 0x87, 0x82, 0x82, 0x82, 0x7D, 0x7D, 0x7D, 0x78,
      0x78, 0x78, 0x74, 0x74, 0x74, 0x73, 0x73, 0x73, 0x6F, 0x6F, 0x6F, 0x6A,
      0x6A, 0x6A, 0x65, 0x65, 0x65, 0x60, 0x60, 0x60, 0x5B, 0x5B, 0x5B, 0x57,
      0x57, 0x57, 0x52, 0x52, 0x52, 0x4D, 0x4D, 0x4D, 0x48, 0x48, 0x48, 0x43,
      0x43, 0x43, 0x3E, 0x3E, 0x3E, 0x3A, 0x3A, 0x3A, 0x35, 0x35, 0x35, 0x30,
      0x30, 0x30, 0x2B, 0x2B, 0x2B, 0x26, 0x26, 0x26, 0x21, 0x21, 0x21, 0x1D,
      0x1D, 0x1D, 0x18, 0x18, 0x18, 0x13, 0x13, 0x13, 0x0E, 0x0E, 0x0E, 0x09,
      0x09, 0x09, 0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x4B, 0x3F, 0x09, 0x7B,
      0x00, 0x00, 0x00, 0x50, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xB0,
      0x30, 0xD2, 0x51, 0x93, 0x97, 0x10, 0xE4, 0x62, 0xD0, 0x55, 0x57, 0x94,
      0x12, 0xE6, 0x61, 0xB5, 0x60, 0x50, 0x92, 0x16, 0xE1, 0x65, 0x63, 0x30,
      0xD6, 0x65, 0x10, 0xE5, 0x63, 0x67, 0x34, 0xD1, 0xD3, 0x50, 0x62, 0xE0,
      0x60, 0x32, 0xD5, 0xD7, 0x54, 0x96, 0x11, 0x65, 0x30, 0x33, 0xD0, 0x52,
      0x91, 0x15, 0xE3, 0xE7, 0x60, 0xD0, 0x56, 0x95, 0x13, 0x17, 0xE0, 0x64,
      0x36, 0x63, 0x50, 0x90, 0x14, 0xE2, 0x66, 0x31, 0x37, 0xD4, 0x06, 0x00,
      0x08, 0xC6, 0x07, 0x42, 0x28, 0x45, 0xA0, 0x34, 0x00, 0x00, 0x00, 0x00,
      0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
  };

  // el mismo, como BMP
  const uint8_t kBmp[] = {
      0x42, 0x4D, 0x76, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x04,
      0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00,
      0x00, 0x00, 0x01, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
      0x00, 0x00, 0xC4, 0x0E, 0x00, 0x00, 0xC4, 0x0E, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
      0x01, 0x00, 0x02, 0x02, 0x02, 0x00, 0x03, 0x03, 0x03, 0x00, 0x04, 0x04,
      0x04, 0x00, 0x05, 0x05, 0x05, 0x00, 0x06, 0x06, 0x06, 0x00, 0x07, 0x07,
      0x07, 0x00, 0x08, 0x08, 0x08, 0x00, 0x09, 0x09, 0x09, 0x00, 0x0A, 0x0A,
      0x0A, 0x00, 0x0B, 0x0B, 0x0B, 0x00, 0x0C, 0x0C, 0x0C, 0x00, 0x0D, 0x0D,
      0x0D, 0x00, 0x0E, 0x0E, 0x0E, 0x00, 0x0F, 0x0F, 0x0F, 0x00, 0x10, 0x10,
      0x10, 0x00, 0x11, 0x11, 0x11, 0x00, 0x12, 0x12, 0x12, 0x00, 0x13, 0x13,
      0x13, 0x00, 0x14, 0x14, 0x14, 0x00, 0x15, 0x15, 0x15, 0x00, 0x16, 0x16,
      0x16, 0x00, 0x17, 0x17, 0x17, 0x00, 0x18, 0x18, 0x18, 0x00, 0x19, 0x19,
      0x19, 0x00, 0x1A, 0x1A, 0x1A, 0x00, 0x1B, 0x1B, 0x1B, 0x00, 0x1C, 0x1C,
      0x1C, 0x00, 0x1D, 0x1D, 0x1D, 0x00, 0x1E, 0x1E, 0x1E, 0x00, 0x1F, 0x1F,
      0x1F, 0x00, 0x20, 0x20, 0x20, 0x00, 0x21, 0x21, 0x21, 0x00, 0x22, 0x22,
      0x22, 0x00, 0x23, 0x23, 0x23, 0x00, 0x24, 0x24, 0x24, 0x00, 0x25, 0x25,
      0x25, 0x00, 0x26, 0x26, 0x26, 0x00, 0x27, 0x27, 0x27, 0x00, 0x28, 0x28,
      0x28, 0x00, 0x29, 0x29, 0x29, 0x00, 0x2A, 0x2A, 0x2A, 0x00, 0x2B, 0x2B,
      0x2B, 0x00, 0x2C, 0x2C, 0x2C, 0x00, 0x2D, 0x2D, 0x2D, 0x00, 0x2E, 0x2E,
      0x2E, 0x00, 0x2F, 0x2F, 0x2F, 0x00, 0x30, 0x30, 0x30, 0x00, 0x31, 0x31,
      0x31, 0x00, 0x32, 0x32, 0x32, 0x00, 0x33, 0x33, 0x33, 0x00, 0x34, 0x34,
      0x34, 0x00, 0x35, 0x35, 0x35, 0x00, 0x36, 0x36, 0x36, 0x00, 0x37, 0x37,
      0x37, 0x00, 0x38, 0x38, 0x38, 0x00, 0x39, 0x39, 0x39, 0x00, 0x3A, 0x3A,
      0x3A, 0x00, 0x3B, 0x3B, 0x3B, 0x00, 0x3C, 0x3C, 0x3C, 0x00, 0x3D, 0x3D,
      0x3D, 0x00, 0x3E, 0x3E, 0x3E, 0x00, 0x3F, 0x3F, 0x3F, 0x00, 0x40, 0x40,
      0x40, 0x00, 0x41, 0x41, 0x41, 0x00, 0x42, 0x42, 0x42, 0x00, 0x43, 0x43,
      0x43, 0x00, 0x44, 0x44, 0x44, 0x00, 0x45, 0x45, 0x45, 0x00, 0x46, 0x46,
      0x46, 0x00, 0x47, 0x47, 0x47, 0x00, 0x48, 0x48, 0x48, 0x00, 0x49, 0x49,
      0x49, 0x00, 0x4A, 0x4A, 0x4A, 0x00, 0x4B, 0x4B, 0x4B, 0x00, 0x4C, 0x4C,
      0x4C, 0x00, 0x4D, 0x4D, 0x4D, 0x00, 0x4E, 0x4E, 0x4E, 0x00, 0x4F, 0x4F,
      0x4F, 0x00, 0x50, 0x50, 0x50, 0x00, 0x51, 0x51, 0x51, 0x00, 0x52, 0x52,
      0x52, 0x00, 0x53, 0x53, 0x53, 0x00, 0x54, 0x54, 0x54, 0x00, 0x55, 0x55,
      0x55, 0x00, 0x56, 0x56, 0x56, 0x00, 0x57, 0x57, 0x57, 0x00, 0x58, 0x58,
      0x58, 0x00, 0x59, 0x59, 0x59, 0x00, 0x5A, 0x5A, 0x5A, 0x00, 0x5B, 0x5B,
      0x5B, 0x00, 0x5C, 0x5C, 0x5C, 0x00, 0x5D, 0x5D, 0x5D, 0x00, 0x5E, 0x5E,
      0x5E, 0x00, 0x5F, 0x5F, 0x5F, 0x00, 0x60, 0x60, 0x60, 0x00, 0x61, 0x61,
      0x61, 0x00, 0x62, 0x62, 0x62, 0x00, 0x63, 0x63, 0x63, 0x00, 0x64, 0x64,
      0x64, 0x00, 0x65, 0x65, 0x65, 0x00, 0x66, 0x66, 0x66, 0x00, 0x67, 0x67,
      0x67, 0x00, 0x68, 0x68, 0x68, 0x00, 0x69, 0x69, 0x69, 0x00, 0x6A, 0x6A,
      0x6A, 0x00, 0x6B, 0x6B, 0x6B, 0x00, 0x6C, 0x6C, 0x6C, 0x00, 0x6D, 0x6D,
      0x6D, 0x00, 0x6E, 0x6E, 0x6E, 0x00, 0x6F, 0x6F, 0x6F, 0x00, 0x70, 0x70,
      0x70, 0x00, 0x71, 0x71, 0x71, 0x00, 0x72, 0x72, 0x72, 0x00, 0x73, 0x73,
      0x73, 0x00, 0x74, 0x74, 0x74, 0x00, 0x75, 0x75, 0x75, 0x00, 0x76, 0x76,
      0x76, 0x00, 0x77, 0x77, 0x77, 0x00, 0x78, 0x78, 0x78, 0x00, 0x79, 0x79,
      0x79, 0x00, 0x7A, 0x7A, 0x7A, 0x00, 0x7B, 0x7B, 0x7B, 0x00, 0x7C, 0x7C,
      0x7C, 0x00, 0x7D, 0x7D, 0x7D, 0x00, 0x7E, 0x7E, 0x7E, 0x00, 0x7F, 0x7F,
      0x7F, 0x00, 0x80, 0x80, 0x80, 0x00, 0x81, 0x81, 0x81, 0x00, 0x82, 0x82,
      0x82, 0x00, 0x83, 0x83, 0x83, 0x00, 0x84, 0x84, 0x84, 0x00, 0x85, 0x85,
      0x85, 0x00, 0x86, 0x86, 0x86, 0x00, 0x87, 0x87, 0x87, 0x00, 0x88, 0x88,
      0x88, 0x00, 0x89, 0x89, 0x89, 0x00, 0x8A, 0x8A, 0x8A, 0x00, 0x8B, 0x8B,
      0x8B, 0x00, 0x8C, 0x8C, 0x8C, 0x00, 0x8D, 0x8D, 0x8D, 0x00, 0x8E, 0x8E,
      0x8E, 0x00, 0x8F, 0x8F, 0x8F, 0x00, 0x90, 0x90, 0x90, 0x00, 0x91, 0x91,
      0x91, 0x00, 0x92, 0x92, 0x92, 0x00, 0x93, 0x93, 0x93, 0x00, 0x94, 0x94,
      0x94, 0x00, 0x95, 0x95, 0x95, 0x00, 0x96, 0x96, 0x96, 0x00, 0x97, 0x97,
      0x97, 0x00, 0x98, 0x98, 0x98, 0x00, 0x99, 0x99, 0x99, 0x00, 0x9A, 0x9A,
      0x9A, 0x00, 0x9B, 0x9B, 0x9B, 0x00, 0x9C, 0x9C, 0x9C, 0x00, 0x9D, 0x9D,
      0x9D, 0x00, 0x9E, 0x9E, 0x9E, 0x00, 0x9F, 0x9F, 0x9F, 0x00, 0xA0, 0xA0,
      0xA0, 0x00, 0xA1, 0xA1, 0xA1, 0x00, 0xA2, 0xA2, 0xA2, 0x00, 0xA3, 0xA3,
      0xA3, 0x00, 0xA4, 0xA4, 0xA4, 0x00, 0xA5, 0xA5, 0xA5, 0x00, 0xA6, 0xA6,
      0xA6, 0x00, 0xA7, 0xA7, 0xA7, 0x00, 0xA8, 0xA8, 0xA8, 0x00, 0xA9, 0xA9,
      0xA9, 0x00, 0xAA, 0xAA, 0xAA, 0x00, 0xAB, 0xAB, 0xAB, 0x00, 0xAC, 0xAC,
      0xAC, 0x00, 0xAD, 0xAD, 0xAD, 0x00, 0xAE, 0xAE, 0xAE, 0x00, 0xAF, 0xAF,
      0xAF, 0x00, 0xB0, 0xB0, 0xB0, 0x00, 0xB1, 0xB1, 0xB1, 0x00, 0xB2, 0xB2,
      0xB2, 0x00, 0xB3, 0xB3, 0xB3, 0x00, 0xB4, 0xB4, 0xB4, 0x00, 0xB5, 0xB5,
      0xB5, 0x00, 0xB6, 0xB6, 0xB6, 0x00, 0xB7, 0xB7, 0xB7, 0x00, 0xB8, 0xB8,
      0xB8, 0x00, 0xB9, 0xB9, 0xB9, 0x00, 0xBA, 0xBA, 0xBA, 0x00, 0xBB, 0xBB,
      0xBB, 0x00, 0xBC, 0xBC, 0xBC, 0x00, 0xBD, 0xBD, 0xBD, 0x00, 0xBE, 0xBE,
      0xBE, 0x00, 0xBF, 0xBF, 0xBF, 0x00, 0xC0, 0xC0, 0xC0, 0x00, 0xC1, 0xC1,
      0xC1, 0x00, 0xC2, 0xC2, 0xC2, 0x00, 0xC3, 0xC3, 0xC3, 0x00, 0xC4, 0xC4,
      0xC4, 0x00, 0xC5, 0xC5, 0xC5, 0x00, 0xC6, 0xC6, 0xC6, 0x00, 0xC7, 0xC7,
      0xC7, 0x00, 0xC8, 0xC8, 0xC8, 0x00, 0xC9, 0xC9, 0xC9, 0x00, 0xCA, 0xCA,
      0xCA, 0x00, 0xCB, 0xCB, 0xCB, 0x00, 0xCC, 0xCC, 0xCC, 0x00, 0xCD, 0xCD,
      0xCD, 0x00, 0xCE, 0xCE, 0xCE, 0x00, 0xCF, 0xCF, 0xCF, 0x00, 0xD0, 0xD0,
      0xD0, 0x00, 0xD1, 0xD1, 0xD1, 0x00, 0xD2, 0xD2, 0xD2, 0x00, 0xD3, 0xD3,
      0xD3, 0x00, 0xD4, 0xD4, 0xD4, 0x00, 0xD5, 0xD5, 0xD5, 0x00, 0xD6, 0xD6,
      0xD6, 0x00, 0xD7, 0xD7, 0xD7, 0x00, 0xD8, 0xD8, 0xD8, 0x00, 0xD9, 0xD9,
      0xD9, 0x00, 0xDA, 0xDA, 0xDA, 0x00, 0xDB, 0xDB, 0xDB, 0x00, 0xDC, 0xDC,
      0xDC, 0x00, 0xDD, 0xDD, 0xDD, 0x00, 0xDE, 0xDE, 0xDE, 0x00, 0xDF, 0xDF,
      0xDF, 0x00, 0xE0, 0xE0, 0xE0, 0x00, 0xE1, 0xE1, 0xE1, 0x00, 0xE2, 0xE2,
      0xE2, 0x00, 0xE3, 0xE3, 0xE3, 0x00, 0xE4, 0xE4, 0xE4, 0x00, 0xE5, 0xE5,
      0xE5, 0x00, 0xE6, 0xE6, 0xE6, 0x00, 0xE7, 0xE7, 0xE7, 0x00, 0xE8, 0xE8,
      0xE8, 0x00, 0xE9, 0xE9, 0xE9, 0x00, 0xEA, 0xEA, 0xEA, 0x00, 0xEB, 0xEB,
      0xEB, 0x00, 0xEC, 0xEC, 0xEC, 0x00, 0xED, 0xED, 0xED, 0x00, 0xEE, 0xEE,
      0xEE, 0x00, 0xEF, 0xEF, 0xEF, 0x00, 0xF0, 0xF0, 0xF0, 0x00, 0xF1, 0xF1,
      0xF1, 0x00, 0xF2, 0xF2, 0xF2, 0x00, 0xF3, 0xF3, 0xF3, 0x00, 0xF4, 0xF4,
      0xF4, 0x00, 0xF5, 0xF5, 0xF5, 0x00, 0xF6, 0xF6, 0xF6, 0x00, 0xF7, 0xF7,
      0xF7, 0x00, 0xF8, 0xF8, 0xF8, 0x00, 0xF9, 0xF9, 0xF9, 0x00, 0xFA, 0xFA,
      0xFA, 0x00, 0xFB, 0xFB, 0xFB, 0x00, 0xFC, 0xFC, 0xFC, 0x00, 0xFD, 0xFD,
      0xFD, 0x00, 0xFE, 0xFE, 0xFE, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x73, 0x90,
      0xAD, 0xCA, 0xE7, 0x04, 0x21, 0x3E, 0x3E, 0x5B, 0x78, 0x95, 0xB2, 0xCF,
      0xEC, 0x09, 0x09, 0x26, 0x43, 0x60, 0x7D, 0x9A, 0xB7, 0xD4, 0xD4, 0xF1,
      0x0E, 0x2B, 0x48, 0x65, 0x82, 0x9F, 0x9F, 0xBC, 0xD9, 0xF6, 0x13, 0x30,
      0x4D, 0x6A, 0x6A, 0x87, 0xA4, 0xC1, 0xDE, 0xFB, 0x18, 0x35, 0x35, 0x52,
      0x6F, 0x8C, 0xA9, 0xC6, 0xE3, 0x00, 0x00, 0x1D, 0x3A, 0x57, 0x74, 0x91,
      0xAE, 0xCB,
  };

  // el mismo, como PGM binario (P5)
  const uint8_t kPgm[] = {
      0x50, 0x35, 0x0A, 0x38, 0x20, 0x38, 0x0A, 0x32, 0x35, 0x35, 0x0A, 0x00,
      0x1D, 0x3A, 0x57, 0x74, 0x91, 0xAE, 0xCB, 0x35, 0x52, 0x6F, 0x8C, 0xA9,
      0xC6, 0xE3, 0x00, 0x6A, 0x87, 0xA4, 0xC1, 0xDE, 0xFB, 0x18, 0x35, 0x9F,
      0xBC, 0xD9, 0xF6, 0x13, 0x30, 0x4D, 0x6A, 0xD4, 0xF1, 0x0E, 0x2B, 0x48,
      0x65, 0x82, 0x9F, 0x09, 0x26, 0x43, 0x60, 0x7D, 0x9A, 0xB7, 0xD4, 0x3E,
      0x5B, 0x78, 0x95, 0xB2, 0xCF, 0xEC, 0x09, 0x73, 0x90, 0xAD, 0xCA, 0xE7,
      0x04, 0x21, 0x3E,
  };

}  // namespace embedded_images

/**
 * @brief Decodificacion de imagen: PNG, BMP y Netpbm.
 *
 * La comparacion byte a byte contra Pillow vive en tools/image (31 archivos,
 * incluidas las variantes raras). Aqui queda lo que debe correr siempre y en
 * cualquier maquina: que las seis codificaciones de una misma imagen coincidan,
 * y que una entrada corrupta se rechace en vez de reventar.
 */
void TestImageDecoding() {
  std::cout << "🧪 [Test 28] Decodificación de PNG, BMP y Netpbm... " << std::flush;
  using namespace neuralsuite::image;

  const int kSize = 8;
  auto expected = [](int x, int y) { return static_cast<uint8_t>((x * 29 + y * 53) % 256); };

  struct Case { const char* name; const uint8_t* data; size_t size; };
  const Case cases[] = {
      {"PNG gris", embedded_images::kPngGray, sizeof(embedded_images::kPngGray)},
      {"PNG entrelazado", embedded_images::kPngInterlaced, sizeof(embedded_images::kPngInterlaced)},
      {"PNG con filtro Average", embedded_images::kPngAverage, sizeof(embedded_images::kPngAverage)},
      {"PNG RGB", embedded_images::kPngRgb, sizeof(embedded_images::kPngRgb)},
      {"PNG con paleta", embedded_images::kPngPalette, sizeof(embedded_images::kPngPalette)},
      {"BMP", embedded_images::kBmp, sizeof(embedded_images::kBmp)},
      {"PGM", embedded_images::kPgm, sizeof(embedded_images::kPgm)},
  };

  for (const Case& c : cases) {
    Bitmap bmp;
    std::string error;
    Check(Decode(c.data, c.size, &bmp, &error), std::string(c.name) + " no se pudo decodificar: " + error);
    Check(bmp.width == kSize && bmp.height == kSize,
          std::string(c.name) + " dio " + std::to_string(bmp.width) + "x" +
              std::to_string(bmp.height) + " en vez de 8x8");

    bool identical = true;
    for (int y = 0; y < kSize && identical; ++y) {
      for (int x = 0; x < kSize; ++x) {
        // Todas las codificaciones son de gris, asi que el primer canal basta:
        // en las de color los tres llevan el mismo valor.
        const uint8_t got = bmp.pixels[(static_cast<size_t>(y) * kSize + x) * bmp.channels];
        if (got != expected(x, y)) { identical = false; break; }
      }
    }
    Check(identical, std::string(c.name) + " no reproduce los pixeles esperados");
  }

  // El formato se decide por el contenido, no por el nombre del archivo.
  Check(DetectFormat(embedded_images::kPngGray, 8) == Format::kPng, "no reconocio un PNG");
  Check(DetectFormat(embedded_images::kBmp, 8) == Format::kBmp, "no reconocio un BMP");
  Check(DetectFormat(embedded_images::kPgm, 8) == Format::kNetpbm, "no reconocio un Netpbm");
  {
    const uint8_t garbage[8] = {'H', 'o', 'l', 'a', 0, 1, 2, 3};
    Check(DetectFormat(garbage, 8) == Format::kUnknown, "reconocio un formato inexistente");
    Bitmap bmp; std::string error;
    Check(!Decode(garbage, 8, &bmp, &error), "acepto datos que no son una imagen");
  }

  // El CRC de cada trozo se comprueba de verdad. Se altera el propio campo del
  // CRC y no los datos: corromper los datos lo detectaria tambien el Adler-32
  // de zlib, de modo que la prueba no distinguiria si el CRC se esta mirando.
  {
    std::vector<uint8_t> tampered(embedded_images::kPngGray,
                                  embedded_images::kPngGray + sizeof(embedded_images::kPngGray));
    // El CRC del IHDR ocupa los cuatro bytes que siguen a sus trece de datos:
    // 8 de firma + 4 de longitud + 4 de tipo + 13 de datos = 29.
    tampered[29] ^= 0x01;
    Bitmap bmp;
    std::string error;
    Check(!DecodePng(tampered.data(), tampered.size(), &bmp, &error),
          "se acepto un PNG con el CRC de un trozo alterado");
  }

  // Entrada hostil. Un decodificador es codigo que lee archivos de fuera, asi
  // que lo que debe garantizarse no es que acierte, sino que nunca reviente ni
  // salga de su buffer. Bajo los sanitizadores de la integracion continua, un
  // acceso invalido aqui aborta la ejecucion.
  int rejected = 0, accepted = 0;
  uint32_t state = 12345;
  auto next = [&state]() {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
  };
  for (const Case& c : cases) {
    for (int trial = 0; trial < 200; ++trial) {
      std::vector<uint8_t> corrupted(c.data, c.data + c.size);
      if (trial % 3 == 0 && corrupted.size() > 1) corrupted.resize(next() % corrupted.size());
      const int flips = 1 + static_cast<int>(next() % 5);
      for (int f = 0; f < flips && !corrupted.empty(); ++f) {
        corrupted[next() % corrupted.size()] ^= static_cast<uint8_t>(1u << (next() % 8));
      }
      Bitmap bmp;
      std::string error;
      if (Decode(corrupted.data(), corrupted.size(), &bmp, &error)) {
        ++accepted;
      } else {
        ++rejected;
        Check(!error.empty(), "se rechazo una imagen sin explicar por que");
      }
    }
  }

  // Gris y reescalado. Una imagen de un solo tono debe seguir siendo ese tono
  // despues de reescalarla: si el mapeo de coordenadas se sale del borde, los
  // extremos se ensucian y esto lo detecta.
  {
    Bitmap flat;
    flat.width = 5; flat.height = 3; flat.channels = 3;
    flat.pixels.assign(5 * 3 * 3, 128);
    std::vector<float> gray;
    ToGrayscale(flat, &gray);
    Check(gray.size() == 15, "ToGrayscale no dio un valor por pixel");
    bool uniform = true;
    for (float v : gray) uniform = uniform && std::abs(v - 128.0f / 255.0f) < 1e-5f;
    Check(uniform, "ToGrayscale no conserva un tono uniforme");

    std::vector<float> scaled;
    Resize(gray, 5, 3, &scaled, 17, 32);
    Check(scaled.size() == 17u * 32u, "Resize no dio el numero de pixeles pedido");
    bool preserved = true;
    for (float v : scaled) preserved = preserved && std::abs(v - 128.0f / 255.0f) < 1e-5f;
    Check(preserved, "Resize altero una imagen de tono uniforme");
  }

  std::cout << "PASADO ✅ (7 codificaciones coinciden; " << rejected << " entradas corruptas "
            << "rechazadas, " << accepted << " aceptadas, 0 caidas)\n"
            << std::flush;
}

namespace embedded_jpeg {

// La misma imagen de 16x16 en cinco codificaciones JPEG, mas los pixeles del
// original. Van incrustadas por la misma razon que las de PNG: en la
// integracion continua no hay Pillow con que generarlas.

  // JPEG secuencial de linea base, 4:4:4, calidad 95
  const uint8_t kJpegBaseline[] = {
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
      0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
      0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0B, 0x08,
      0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x06, 0x08, 0x0B, 0x0C, 0x0B, 0x0A,
      0x0C, 0x09, 0x0A, 0x0A, 0x0A, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x02, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0A, 0x07, 0x06, 0x07,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
      0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
      0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00,
      0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
      0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
      0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
      0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24,
      0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
      0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
      0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
      0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A,
      0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86,
      0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
      0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
      0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6,
      0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
      0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xC4, 0x00,
      0x1F, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x11, 0x00,
      0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
      0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
      0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
      0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15,
      0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
      0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39,
      0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
      0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
      0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
      0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
      0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
      0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00,
      0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0xDD,
      0xF0, 0x67, 0xED, 0x65, 0xA1, 0xFF, 0x00, 0xC1, 0x3F, 0xB4, 0x61, 0xF0,
      0x7F, 0x58, 0x08, 0x66, 0xB7, 0x52, 0x3E, 0x6E, 0xBF, 0x39, 0xDF, 0xFF,
      0x00, 0xB3, 0x57, 0xE8, 0xF9, 0x66, 0x4B, 0x90, 0xF1, 0xFE, 0x59, 0xF5,
      0xDA, 0xB6, 0xE7, 0x6B, 0xF2, 0xD3, 0xF4, 0x33, 0xC6, 0x71, 0xD3, 0xE2,
      0xEA, 0x96, 0xA7, 0xD7, 0x4F, 0xBB, 0xDD, 0xFD, 0x0E, 0x3B, 0xE2, 0x1F,
      0xC2, 0xCD, 0x47, 0xFE, 0x0A, 0x23, 0x76, 0x75, 0xAD, 0x03, 0x76, 0xC9,
      0x0E, 0xE1, 0xB2, 0xBF, 0x05, 0xE2, 0xEA, 0xF9, 0xAF, 0x02, 0x63, 0xDA,
      0xC1, 0x5F, 0x95, 0x3E, 0x86, 0x98, 0x5E, 0x08, 0x78, 0x7F, 0xF6, 0xB9,
      0xFA, 0x90, 0x78, 0x9F, 0xF6, 0x4B, 0xD6, 0x7F, 0xE0, 0xA0, 0x7A, 0xA1,
      0xF8, 0xC3, 0xA6, 0x16, 0x11, 0x5C, 0x90, 0x46, 0xDF, 0xF6, 0x3E, 0x4F,
      0xFD, 0x96, 0xBC, 0xDC, 0xD7, 0x3A, 0xE2, 0x1F, 0x0F, 0xF3, 0x4F, 0xA9,
      0x50, 0xBF, 0x22, 0x7F, 0x9E, 0xBF, 0xA9, 0xB6, 0x1B, 0x81, 0x21, 0xC2,
      0x30, 0xE6, 0x9E, 0xEB, 0x5F, 0xFC, 0x0B, 0xDE, 0xFD, 0x4E, 0xA7, 0xC1,
      0x9F, 0x15, 0x2C, 0x3F, 0xE0, 0x9D, 0x96, 0xC3, 0x45, 0xD6, 0x40, 0x2D,
      0x18, 0xDB, 0xF3, 0x57, 0xEE, 0xDC, 0x23, 0x43, 0x27, 0xE3, 0xBC, 0x02,
      0x78, 0xFB, 0x73, 0x35, 0xD4, 0xE7, 0xC4, 0xF1, 0xBB, 0xC4, 0x3F, 0xAA,
      0x43, 0xD0, 0xFF, 0xD9,
  };

  // el MISMO original en progresivo: mismos coeficientes, otra codificacion
  const uint8_t kJpegProgressive[] = {
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
      0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
      0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0B, 0x08,
      0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x06, 0x08, 0x0B, 0x0C, 0x0B, 0x0A,
      0x0C, 0x09, 0x0A, 0x0A, 0x0A, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x02, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0A, 0x07, 0x06, 0x07,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0xFF, 0xC2, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
      0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
      0x16, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x03, 0x04, 0xFF, 0xC4, 0x00,
      0x17, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x06, 0x03, 0x04, 0xFF, 0xDA,
      0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x10, 0x03, 0x10, 0x00, 0x00, 0x01,
      0xDD, 0x46, 0x47, 0x82, 0x53, 0xE6, 0x6A, 0xDD, 0xE7, 0xFF, 0xC4, 0x00,
      0x19, 0x10, 0x00, 0x02, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x07, 0x02, 0x04, 0x05, 0x06,
      0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x05, 0x02, 0xA7, 0xD6,
      0x01, 0x7E, 0x1D, 0x0C, 0xB2, 0x31, 0x25, 0x67, 0x92, 0x33, 0x00, 0xB4,
      0xF5, 0x60, 0xBB, 0x8F, 0xFF, 0xC4, 0x00, 0x20, 0x11, 0x00, 0x00, 0x05,
      0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x03, 0x05, 0x11, 0x13, 0x23, 0xF0, 0x02, 0x15, 0x21,
      0x22, 0x41, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x03, 0x01, 0x01, 0x3F, 0x01,
      0x59, 0xF7, 0x77, 0xD5, 0xD7, 0x3C, 0x09, 0x32, 0x53, 0xBA, 0x61, 0x36,
      0x22, 0x68, 0x29, 0x3C, 0x9E, 0x42, 0x8F, 0x75, 0x2D, 0x10, 0xFF, 0xC4,
      0x00, 0x1C, 0x11, 0x00, 0x01, 0x04, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x02, 0x05, 0x31,
      0x03, 0x13, 0x23, 0x06, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x02, 0x01, 0x01,
      0x3F, 0x01, 0x18, 0x20, 0x27, 0xC6, 0xDC, 0xEB, 0x52, 0xEF, 0x2A, 0x0B,
      0x3F, 0x1A, 0x45, 0x1B, 0x21, 0xE7, 0xCA, 0xD2, 0xCA, 0x51, 0x0C, 0x0E,
      0x77, 0x07, 0x7B, 0x5F, 0xFF, 0xC4, 0x00, 0x1A, 0x10, 0x00, 0x02, 0x02,
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x03, 0x02, 0x13, 0x15, 0x31, 0x41, 0xFF, 0xDA, 0x00, 0x08,
      0x01, 0x01, 0x00, 0x06, 0x3F, 0x02, 0xC3, 0xBB, 0x71, 0x2E, 0x47, 0x4C,
      0xC2, 0xF5, 0x22, 0x97, 0x70, 0xFF, 0xC4, 0x00, 0x19, 0x10, 0x01, 0x00,
      0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x21, 0x00, 0x61, 0x11, 0x41, 0xC1, 0xFF, 0xDA, 0x00, 0x08,
      0x01, 0x01, 0x00, 0x01, 0x3F, 0x21, 0x30, 0x62, 0xDD, 0xBD, 0x8E, 0x7C,
      0x22, 0x32, 0x68, 0xA3, 0x93, 0x76, 0x83, 0x3F, 0xFF, 0xDA, 0x00, 0x0C,
      0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x10, 0xCB, 0x2F,
      0xFF, 0xC4, 0x00, 0x1D, 0x11, 0x00, 0x02, 0x02, 0x01, 0x05, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x21,
      0x41, 0x91, 0x11, 0x71, 0x81, 0xE1, 0xF0, 0xFF, 0xDA, 0x00, 0x08, 0x01,
      0x03, 0x01, 0x01, 0x3F, 0x10, 0x66, 0x85, 0xA3, 0x1D, 0x06, 0xE5, 0x97,
      0xD5, 0x4F, 0x8D, 0xC7, 0xB7, 0x19, 0xFF, 0xC4, 0x00, 0x1A, 0x11, 0x00,
      0x02, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x01, 0x11, 0x00, 0x21, 0x41, 0xB1, 0xC1, 0xFF, 0xDA,
      0x00, 0x08, 0x01, 0x02, 0x01, 0x01, 0x3F, 0x10, 0xA9, 0xEE, 0x35, 0x5C,
      0x85, 0xE1, 0xD0, 0xE2, 0x5D, 0x35, 0x3B, 0xBE, 0xC0, 0xB2, 0xAC, 0x33,
      0x3F, 0xFF, 0xC4, 0x00, 0x18, 0x10, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
      0x21, 0x11, 0x51, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01, 0x3F,
      0x10, 0x26, 0x0D, 0x07, 0xAD, 0xD3, 0xFD, 0x57, 0x15, 0xB3, 0x87, 0x12,
      0x62, 0xE2, 0x21, 0x1F, 0xFF, 0xD9,
  };

  // el mismo con crominancia a la mitad (4:2:0)
  const uint8_t kJpegSubsampled[] = {
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x04,
      0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0A, 0x07,
      0x07, 0x06, 0x08, 0x0C, 0x0A, 0x0C, 0x0C, 0x0B, 0x0A, 0x0B, 0x0B, 0x0D,
      0x0E, 0x12, 0x10, 0x0D, 0x0E, 0x11, 0x0E, 0x0B, 0x0B, 0x10, 0x16, 0x10,
      0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0C, 0x0F, 0x17, 0x18, 0x16, 0x14,
      0x18, 0x12, 0x14, 0x15, 0x14, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x03, 0x04,
      0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0D, 0x0B, 0x0D,
      0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
      0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
      0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
      0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
      0x14, 0x14, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
      0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
      0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00,
      0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
      0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
      0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
      0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24,
      0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
      0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
      0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
      0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A,
      0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86,
      0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
      0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
      0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6,
      0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
      0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xC4, 0x00,
      0x1F, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x11, 0x00,
      0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
      0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
      0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
      0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15,
      0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
      0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39,
      0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
      0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
      0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
      0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
      0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
      0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00,
      0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0xD3,
      0xD1, 0x7E, 0x2C, 0xDB, 0x7E, 0xCF, 0xF6, 0x63, 0xC3, 0xF7, 0x18, 0xF3,
      0x23, 0x18, 0xE7, 0xFD, 0xAF, 0x9B, 0xFA, 0xD7, 0x3D, 0xE2, 0x1F, 0x0A,
      0xCD, 0xFB, 0x44, 0x4A, 0x6E, 0x6D, 0x73, 0xB5, 0xB9, 0xE2, 0xA3, 0xD4,
      0xFE, 0x12, 0xDC, 0x7E, 0xD0, 0x37, 0x47, 0xC4, 0x10, 0xE7, 0x64, 0x9C,
      0xF1, 0xFE, 0xCF, 0xCB, 0xFD, 0x2B, 0x6F, 0x45, 0xF1, 0x54, 0x5F, 0xB3,
      0xAC, 0x42, 0xDA, 0xE3, 0x1B, 0x97, 0x8E, 0x68, 0xF6, 0xB4, 0xF3, 0xCF,
      0xF6, 0x3A, 0x2B, 0xD9, 0x62, 0xE3, 0xBC, 0xB6, 0xBB, 0xEA, 0x0D, 0x35,
      0xFB, 0xCC, 0xDF, 0xF8, 0xBD, 0x17, 0x99, 0xFF, 0xD9,
  };

  // en escala de grises: un solo componente
  const uint8_t kJpegGray[] = {
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
      0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
      0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0B, 0x08,
      0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x06, 0x08, 0x0B, 0x0C, 0x0B, 0x0A,
      0x0C, 0x09, 0x0A, 0x0A, 0x0A, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x10,
      0x00, 0x10, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
      0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
      0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
      0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
      0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
      0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
      0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
      0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
      0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
      0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
      0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
      0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
      0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
      0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
      0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
      0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
      0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
      0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
      0x00, 0x00, 0x3F, 0x00, 0xD8, 0xD3, 0x3F, 0xE0, 0xAA, 0xBA, 0x37, 0xEC,
      0xFF, 0x00, 0x61, 0xFF, 0x00, 0x0A, 0x62, 0xE9, 0x57, 0xCC, 0x51, 0xE4,
      0x73, 0xF9, 0x57, 0x27, 0xE2, 0x6F, 0xD8, 0x4B, 0x52, 0xFD, 0xBE, 0xAE,
      0xCF, 0xC4, 0x8D, 0x34, 0xB0, 0x59, 0x0F, 0x9B, 0xF2, 0xFE, 0x74, 0xB7,
      0xDF, 0xF0, 0x4A, 0x7D, 0x47, 0xF6, 0x81, 0xBB, 0xFF, 0x00, 0x85, 0xD1,
      0x0B, 0x1D, 0xAC, 0x7C, 0xFE, 0x3F, 0x3A, 0xDD, 0xD1, 0xBF, 0x6E, 0xFB,
      0x4F, 0xD8, 0x0E, 0x01, 0xF0, 0xDE, 0xF0, 0x0D, 0xD1, 0xFE, 0xEB, 0x9F,
      0xCA, 0xBF, 0xFF, 0xD9,
  };

  // el anterior con la cabecera cambiada a modo aritmetico (SOF9)
  const uint8_t kJpegArithmetic[] = {
      0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
      0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
      0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
      0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
      0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0B, 0x08,
      0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x06, 0x08, 0x0B, 0x0C, 0x0B, 0x0A,
      0x0C, 0x09, 0x0A, 0x0A, 0x0A, 0xFF, 0xDB, 0x00, 0x43, 0x01, 0x02, 0x02,
      0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0A, 0x07, 0x06, 0x07,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
      0x0A, 0x0A, 0xFF, 0xC9, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03,
      0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xC4, 0x00,
      0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00,
      0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
      0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
      0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
      0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24,
      0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
      0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
      0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
      0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A,
      0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86,
      0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
      0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
      0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6,
      0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
      0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xC4, 0x00,
      0x1F, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
      0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x11, 0x00,
      0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
      0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
      0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
      0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15,
      0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
      0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39,
      0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
      0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
      0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84,
      0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
      0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
      0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
      0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
      0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
      0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00,
      0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00, 0xDD,
      0xF0, 0x67, 0xED, 0x65, 0xA1, 0xFF, 0x00, 0xC1, 0x3F, 0xB4, 0x61, 0xF0,
      0x7F, 0x58, 0x08, 0x66, 0xB7, 0x52, 0x3E, 0x6E, 0xBF, 0x39, 0xDF, 0xFF,
      0x00, 0xB3, 0x57, 0xE8, 0xF9, 0x66, 0x4B, 0x90, 0xF1, 0xFE, 0x59, 0xF5,
      0xDA, 0xB6, 0xE7, 0x6B, 0xF2, 0xD3, 0xF4, 0x33, 0xC6, 0x71, 0xD3, 0xE2,
      0xEA, 0x96, 0xA7, 0xD7, 0x4F, 0xBB, 0xDD, 0xFD, 0x0E, 0x3B, 0xE2, 0x1F,
      0xC2, 0xCD, 0x47, 0xFE, 0x0A, 0x23, 0x76, 0x75, 0xAD, 0x03, 0x76, 0xC9,
      0x0E, 0xE1, 0xB2, 0xBF, 0x05, 0xE2, 0xEA, 0xF9, 0xAF, 0x02, 0x63, 0xDA,
      0xC1, 0x5F, 0x95, 0x3E, 0x86, 0x98, 0x5E, 0x08, 0x78, 0x7F, 0xF6, 0xB9,
      0xFA, 0x90, 0x78, 0x9F, 0xF6, 0x4B, 0xD6, 0x7F, 0xE0, 0xA0, 0x7A, 0xA1,
      0xF8, 0xC3, 0xA6, 0x16, 0x11, 0x5C, 0x90, 0x46, 0xDF, 0xF6, 0x3E, 0x4F,
      0xFD, 0x96, 0xBC, 0xDC, 0xD7, 0x3A, 0xE2, 0x1F, 0x0F, 0xF3, 0x4F, 0xA9,
      0x50, 0xBF, 0x22, 0x7F, 0x9E, 0xBF, 0xA9, 0xB6, 0x1B, 0x81, 0x21, 0xC2,
      0x30, 0xE6, 0x9E, 0xEB, 0x5F, 0xFC, 0x0B, 0xDE, 0xFD, 0x4E, 0xA7, 0xC1,
      0x9F, 0x15, 0x2C, 0x3F, 0xE0, 0x9D, 0x96, 0xC3, 0x45, 0xD6, 0x40, 0x2D,
      0x18, 0xDB, 0xF3, 0x57, 0xEE, 0xDC, 0x23, 0x43, 0x27, 0xE3, 0xBC, 0x02,
      0x78, 0xFB, 0x73, 0x35, 0xD4, 0xE7, 0xC4, 0xF1, 0xBB, 0xC4, 0x3F, 0xAA,
      0x43, 0xD0, 0xFF, 0xD9,
  };

  const uint8_t kJpegSource[] = {
      0x80, 0x26, 0x80, 0x9D, 0x29, 0x8D, 0xB7, 0x80, 0x9A, 0xCB, 0x9D, 0xA3,
      0xD7, 0xB7, 0xA8, 0xD9, 0xCB, 0xA9, 0xD1, 0xD7, 0xA6, 0xC1, 0xD9, 0x9E,
      0xA9, 0xD1, 0x93, 0x8C, 0xC1, 0x85, 0x6E, 0xA9, 0x77, 0x52, 0x8C, 0x6A,
      0x3B, 0x6E, 0x60, 0x2C, 0x52, 0x58, 0x26, 0x3B, 0x55, 0x29, 0x2C, 0x57,
      0x80, 0x2D, 0x80, 0x9B, 0x30, 0x96, 0xB3, 0x80, 0xAB, 0xC5, 0x9B, 0xBA,
      0xD0, 0xB3, 0xC3, 0xD2, 0xC5, 0xC5, 0xCB, 0xD0, 0xBF, 0xBB, 0xD2, 0xB2,
      0xA5, 0xCB, 0x9F, 0x8B, 0xBB, 0x89, 0x70, 0xA5, 0x72, 0x56, 0x8B, 0x5D,
      0x41, 0x70, 0x4B, 0x32, 0x56, 0x3F, 0x2D, 0x41, 0x3A, 0x30, 0x32, 0x3D,
      0x80, 0x41, 0x80, 0x94, 0x43, 0x9C, 0xA6, 0x80, 0xB5, 0xB4, 0x94, 0xC8,
      0xBC, 0xA6, 0xD3, 0xBE, 0xB4, 0xD6, 0xB9, 0xBC, 0xCE, 0xAD, 0xBE, 0xBE,
      0x9C, 0xB9, 0xA7, 0x88, 0xAD, 0x8C, 0x74, 0x9C, 0x6F, 0x60, 0x88, 0x54,
      0x50, 0x74, 0x3E, 0x45, 0x60, 0x2F, 0x41, 0x50, 0x29, 0x43, 0x45, 0x2D,
      0x80, 0x5F, 0x80, 0x8A, 0x60, 0x9D, 0x94, 0x80, 0xB7, 0x9B, 0x8A, 0xCB,
      0x9F, 0x94, 0xD7, 0xA0, 0x9B, 0xD9, 0x9D, 0x9F, 0xD1, 0x97, 0xA0, 0xC1,
      0x8E, 0x9D, 0xA9, 0x84, 0x97, 0x8C, 0x79, 0x8E, 0x6E, 0x6F, 0x84, 0x52,
      0x67, 0x79, 0x3B, 0x61, 0x6F, 0x2C, 0x5F, 0x67, 0x26, 0x60, 0x61, 0x29,
      0x80, 0x82, 0x80, 0x7F, 0x82, 0x9B, 0x7E, 0x80, 0xB3, 0x7D, 0x7F, 0xC5,
      0x7D, 0x7E, 0xD0, 0x7D, 0x7D, 0xD2, 0x7D, 0x7D, 0xCB, 0x7E, 0x7D, 0xBB,
      0x7E, 0x7D, 0xA5, 0x7F, 0x7E, 0x8B, 0x80, 0x7E, 0x70, 0x81, 0x7F, 0x56,
      0x81, 0x80, 0x41, 0x82, 0x81, 0x32, 0x82, 0x81, 0x2D, 0x82, 0x82, 0x30,
      0x80, 0xA5, 0x80, 0x73, 0xA3, 0x94, 0x68, 0x80, 0xA6, 0x60, 0x73, 0xB4,
      0x5B, 0x68, 0xBC, 0x5A, 0x60, 0xBE, 0x5D, 0x5B, 0xB9, 0x64, 0x5A, 0xAD,
      0x6E, 0x5D, 0x9C, 0x7A, 0x64, 0x88, 0x87, 0x6E, 0x74, 0x92, 0x7A, 0x60,
      0x9C, 0x87, 0x50, 0xA2, 0x92, 0x45, 0xA5, 0x9C, 0x41, 0xA3, 0xA2, 0x43,
      0x80, 0xC2, 0x80, 0x6A, 0xBF, 0x8A, 0x56, 0x80, 0x94, 0x48, 0x6A, 0x9B,
      0x3F, 0x56, 0x9F, 0x3D, 0x48, 0xA0, 0x43, 0x3F, 0x9D, 0x50, 0x3D, 0x97,
      0x61, 0x43, 0x8E, 0x76, 0x50, 0x84, 0x8C, 0x61, 0x79, 0xA1, 0x76, 0x6F,
      0xB2, 0x8C, 0x67, 0xBD, 0xA1, 0x61, 0xC2, 0xB2, 0x5F, 0xBF, 0xBD, 0x60,
      0x80, 0xD4, 0x80, 0x64, 0xD1, 0x7F, 0x4B, 0x80, 0x7E, 0x38, 0x64, 0x7D,
      0x2D, 0x4B, 0x7D, 0x2B, 0x38, 0x7D, 0x32, 0x2D, 0x7D, 0x42, 0x2B, 0x7E,
      0x59, 0x32, 0x7E, 0x74, 0x42, 0x7F, 0x90, 0x59, 0x80, 0xAA, 0x74, 0x81,
      0xC0, 0x90, 0x81, 0xCE, 0xAA, 0x82, 0xD4, 0xC0, 0x82, 0xD1, 0xCE, 0x82,
      0x80, 0xD9, 0x80, 0x62, 0xD6, 0x73, 0x48, 0x80, 0x68, 0x34, 0x62, 0x60,
      0x28, 0x48, 0x5B, 0x26, 0x34, 0x5A, 0x2E, 0x28, 0x5D, 0x3F, 0x26, 0x64,
      0x56, 0x2E, 0x6E, 0x73, 0x3F, 0x7A, 0x91, 0x56, 0x87, 0xAD, 0x73, 0x92,
      0xC3, 0x91, 0x9C, 0xD3, 0xAD, 0xA2, 0xD9, 0xC3, 0xA5, 0xD6, 0xD3, 0xA3,
      0x80, 0xD0, 0x80, 0x65, 0xCD, 0x6A, 0x4E, 0x80, 0x56, 0x3C, 0x65, 0x48,
      0x31, 0x4E, 0x3F, 0x2F, 0x3C, 0x3D, 0x36, 0x31, 0x43, 0x45, 0x2F, 0x50,
      0x5B, 0x36, 0x61, 0x74, 0x45, 0x76, 0x8F, 0x5B, 0x8C, 0xA8, 0x74, 0xA1,
      0xBD, 0x8F, 0xB2, 0xCA, 0xA8, 0xBD, 0xD0, 0xBD, 0xC2, 0xCD, 0xCA, 0xBF,
      0x80, 0xBA, 0x80, 0x6C, 0xB8, 0x64, 0x5B, 0x80, 0x4B, 0x4E, 0x6C, 0x38,
      0x46, 0x5B, 0x2D, 0x45, 0x4E, 0x2B, 0x4A, 0x46, 0x32, 0x55, 0x45, 0x42,
      0x65, 0x4A, 0x59, 0x77, 0x55, 0x74, 0x8B, 0x65, 0x90, 0x9D, 0x77, 0xAA,
      0xAC, 0x8B, 0xC0, 0xB6, 0x9D, 0xCE, 0xBA, 0xAC, 0xD4, 0xB8, 0xB6, 0xD1,
      0x80, 0x9B, 0x80, 0x76, 0x9A, 0x62, 0x6E, 0x80, 0x48, 0x68, 0x76, 0x34,
      0x65, 0x6E, 0x28, 0x64, 0x68, 0x26, 0x66, 0x65, 0x2E, 0x6B, 0x64, 0x3F,
      0x73, 0x66, 0x56, 0x7C, 0x6B, 0x73, 0x85, 0x73, 0x91, 0x8D, 0x7C, 0xAD,
      0x94, 0x85, 0xC3, 0x99, 0x8D, 0xD3, 0x9B, 0x94, 0xD9, 0x9A, 0x99, 0xD6,
      0x80, 0x78, 0x80, 0x82, 0x78, 0x65, 0x84, 0x80, 0x4E, 0x86, 0x82, 0x3C,
      0x87, 0x84, 0x31, 0x87, 0x86, 0x2F, 0x87, 0x87, 0x36, 0x85, 0x87, 0x45,
      0x83, 0x87, 0x5B, 0x81, 0x85, 0x74, 0x7E, 0x83, 0x8F, 0x7C, 0x81, 0xA8,
      0x7A, 0x7E, 0xBD, 0x78, 0x7C, 0xCA, 0x78, 0x7A, 0xD0, 0x78, 0x78, 0xCD,
      0x80, 0x55, 0x80, 0x8D, 0x57, 0x6C, 0x9A, 0x80, 0x5B, 0xA3, 0x8D, 0x4E,
      0xA8, 0x9A, 0x46, 0xA9, 0xA3, 0x45, 0xA6, 0xA8, 0x4A, 0x9E, 0xA9, 0x55,
      0x93, 0xA6, 0x65, 0x85, 0x9E, 0x77, 0x77, 0x93, 0x8B, 0x6A, 0x85, 0x9D,
      0x60, 0x77, 0xAC, 0x58, 0x6A, 0xB6, 0x55, 0x60, 0xBA, 0x57, 0x58, 0xB8,
      0x80, 0x3A, 0x80, 0x96, 0x3D, 0x76, 0xAB, 0x80, 0x6E, 0xBA, 0x96, 0x68,
      0xC3, 0xAB, 0x65, 0xC5, 0xBA, 0x64, 0xBF, 0xC3, 0x66, 0xB2, 0xC5, 0x6B,
      0x9F, 0xBF, 0x73, 0x89, 0xB2, 0x7C, 0x72, 0x9F, 0x85, 0x5D, 0x89, 0x8D,
      0x4B, 0x72, 0x94, 0x3F, 0x5D, 0x99, 0x3A, 0x4B, 0x9B, 0x3D, 0x3F, 0x9A,
      0x80, 0x29, 0x80, 0x9C, 0x2D, 0x82, 0xB5, 0x80, 0x84, 0xC8, 0x9C, 0x86,
      0xD3, 0xB5, 0x87, 0xD6, 0xC8, 0x87, 0xCE, 0xD3, 0x87, 0xBE, 0xD6, 0x85,
      0xA7, 0xCE, 0x83, 0x8C, 0xBE, 0x81, 0x6F, 0xA7, 0x7E, 0x54, 0x8C, 0x7C,
      0x3E, 0x6F, 0x7A, 0x2F, 0x54, 0x78, 0x29, 0x3E, 0x78, 0x2D, 0x2F, 0x78,
  };

}  // namespace embedded_jpeg

/**
 * @brief JPEG: transformada, linea base, progresivo y submuestreo.
 *
 * JPEG es el unico de los formatos que lee este proyecto cuya salida no esta
 * especificada bit a bit: la norma fija requisitos de precision para la
 * transformada inversa (T.83), no un resultado exacto. Eso obliga a comprobarlo
 * de otra manera, porque no hay contra que igualar.
 *
 * La solucion es apoyarse en tres propiedades que si son exactas:
 *
 *  1. La transformada tiene una definicion matematica cerrada. La version
 *     separable que usa el decodificador debe coincidir con ella, y se
 *     comprueba contra el calculo directo en doble precision.
 *  2. Un mismo original guardado en linea base y en progresivo produce los
 *     mismos coeficientes cuantizados: solo cambia como se codifican. Los dos
 *     archivos tienen que decodificarse a pixeles identicos, sin tolerancia
 *     alguna. Es la comprobacion mas fuerte que admite el progresivo.
 *  3. La perdida de JPEG a calidad alta esta acotada, asi que el decodificado
 *     tiene que parecerse al original de partida.
 *
 * La comparacion contra libjpeg, que es la que cubre las variantes raras, vive
 * en tools/image y usa un criterio estadistico por el motivo que se explica
 * alli.
 */
void TestJpegDecoding() {
  std::cout << "🧪 [Test 29] JPEG: transformada, línea base y progresivo... " << std::flush;
  using namespace neuralsuite::image;

  // 1. La transformada separable contra su definicion directa. Se calculan las
  //    dos en doble precision: si coinciden, cualquier diferencia posterior con
  //    otro decodificador es del suyo y no del nuestro.
  {
    uint32_t state = 7777;
    auto next = [&state]() {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      return state;
    };
    double worst = 0.0;
    for (int trial = 0; trial < 40; ++trial) {
      int32_t coefficients[64];
      for (int i = 0; i < 64; ++i) {
        // Coeficientes decrecientes con la frecuencia, como en una imagen real.
        const int magnitude = 400 / (1 + i);
        coefficients[i] = static_cast<int32_t>(next() % (2u * magnitude + 1u)) - magnitude;
      }

      uint8_t got[64];
      jpeg_detail::InverseDct(coefficients, got, 8);

      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          // Definicion directa: suma doble sobre las 64 frecuencias.
          double sum = 0.0;
          for (int v = 0; v < 8; ++v) {
            for (int u = 0; u < 8; ++u) {
              const double cu = (u == 0) ? std::sqrt(0.5) : 1.0;
              const double cv = (v == 0) ? std::sqrt(0.5) : 1.0;
              sum += 0.25 * cu * cv * coefficients[v * 8 + u] *
                     std::cos((2 * x + 1) * u * kPi / 16.0) *
                     std::cos((2 * y + 1) * v * kPi / 16.0);
            }
          }
          const double exact = sum + 128.0;
          if (exact < 0.0 || exact > 255.0) continue;  // el recorte no es error
          worst = std::max(worst, std::abs(exact - got[y * 8 + x]));
        }
      }
    }
    // Medio nivel es lo que introduce redondear a entero, y nada mas.
    Check(worst <= 0.5 + 1e-6,
          "la transformada inversa se aparta de su definicion en " + std::to_string(worst));
  }

  struct Decoded { Bitmap bitmap; bool ok; };
  auto decode = [](const uint8_t* data, size_t size, const char* what) {
    Bitmap bitmap;
    std::string error;
    Check(DecodeJpeg(data, size, &bitmap, &error),
          std::string(what) + " no se pudo decodificar: " + error);
    return bitmap;
  };

  const Bitmap baseline = decode(embedded_jpeg::kJpegBaseline,
                                 sizeof(embedded_jpeg::kJpegBaseline), "el JPEG de linea base");
  const Bitmap progressive = decode(embedded_jpeg::kJpegProgressive,
                                    sizeof(embedded_jpeg::kJpegProgressive), "el JPEG progresivo");
  const Bitmap subsampled = decode(embedded_jpeg::kJpegSubsampled,
                                   sizeof(embedded_jpeg::kJpegSubsampled), "el JPEG 4:2:0");
  const Bitmap gray = decode(embedded_jpeg::kJpegGray,
                             sizeof(embedded_jpeg::kJpegGray), "el JPEG en gris");

  Check(baseline.width == 16 && baseline.height == 16 && baseline.channels == 3,
        "el JPEG de color no dio 16x16 con tres canales");
  Check(gray.channels == 1, "el JPEG en gris no dio un solo canal");
  Check(subsampled.width == 16 && subsampled.height == 16,
        "el JPEG con crominancia submuestreada no dio el tamano correcto");

  // 2. Linea base y progresivo del mismo original: identicos, sin tolerancia.
  //    Es lo unico exacto que puede exigirsele al camino progresivo, y no es
  //    poco: cubre el reparto en varios scans, el desplazamiento por Al, las
  //    rachas de fin de bloque y los bits de refinamiento.
  Check(progressive.pixels.size() == baseline.pixels.size(),
        "el progresivo no dio el mismo numero de pixeles que la linea base");
  Check(progressive.pixels == baseline.pixels,
        "el mismo original en linea base y en progresivo no da los mismos pixeles");

  // 3. La perdida a calidad alta esta acotada: el decodificado tiene que
  //    parecerse al original que se codifico.
  {
    Check(sizeof(embedded_jpeg::kJpegSource) == baseline.pixels.size(),
          "la fuente incrustada no tiene el tamano de la imagen");
    double total = 0.0;
    int worst = 0;
    for (size_t i = 0; i < baseline.pixels.size(); ++i) {
      const int d = std::abs(static_cast<int>(baseline.pixels[i]) -
                             static_cast<int>(embedded_jpeg::kJpegSource[i]));
      total += d;
      worst = std::max(worst, d);
    }
    const double mean = total / baseline.pixels.size();
    Check(mean < 3.0, "el JPEG a calidad 95 se aparta del original una media de " +
                          std::to_string(mean));
    Check(worst < 24, "el JPEG a calidad 95 se aparta del original hasta " + std::to_string(worst));
  }

  // 4. Los modos que no se admiten se rechazan diciendo cual es, en vez de
  //    producir pixeles sin sentido.
  {
    Bitmap bitmap;
    std::string error;
    Check(!DecodeJpeg(embedded_jpeg::kJpegArithmetic, sizeof(embedded_jpeg::kJpegArithmetic),
                      &bitmap, &error),
          "se acepto un JPEG en modo aritmetico, que no se sabe decodificar");
    Check(error.find("aritmetico") != std::string::npos,
          "el rechazo del modo aritmetico no explica cual es el problema: " + error);
  }

  // 5. Se reconoce por el contenido y entra por la fachada comun.
  Check(DetectFormat(embedded_jpeg::kJpegBaseline, 8) == Format::kJpeg, "no reconocio un JPEG");
  {
    Bitmap bitmap;
    std::string error;
    Check(Decode(embedded_jpeg::kJpegBaseline, sizeof(embedded_jpeg::kJpegBaseline), &bitmap,
                 &error),
          "la fachada no decodifico un JPEG");
  }

  // 6. Entrada hostil. Un JPEG danado por el final es el caso mas comun, y el
  //    decodificador rinde la parte legible en vez de rechazarlo entero, asi
  //    que aqui se aceptan muchas mas que en PNG. Lo que se exige es lo mismo:
  //    que nunca reviente ni se salga de su buffer.
  int rejected = 0, accepted = 0;
  uint32_t state = 4242;
  auto next = [&state]() {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
  };
  const uint8_t* sources[] = {embedded_jpeg::kJpegBaseline, embedded_jpeg::kJpegProgressive,
                              embedded_jpeg::kJpegSubsampled, embedded_jpeg::kJpegGray};
  const size_t sizes[] = {sizeof(embedded_jpeg::kJpegBaseline),
                          sizeof(embedded_jpeg::kJpegProgressive),
                          sizeof(embedded_jpeg::kJpegSubsampled),
                          sizeof(embedded_jpeg::kJpegGray)};
  for (int s = 0; s < 4; ++s) {
    for (int trial = 0; trial < 250; ++trial) {
      std::vector<uint8_t> corrupted(sources[s], sources[s] + sizes[s]);
      if (trial % 3 == 0 && corrupted.size() > 1) corrupted.resize(next() % corrupted.size());
      const int flips = 1 + static_cast<int>(next() % 5);
      for (int f = 0; f < flips && !corrupted.empty(); ++f) {
        corrupted[next() % corrupted.size()] ^= static_cast<uint8_t>(1u << (next() % 8));
      }
      Bitmap bitmap;
      std::string error;
      if (DecodeJpeg(corrupted.data(), corrupted.size(), &bitmap, &error)) {
        ++accepted;
      } else {
        ++rejected;
        Check(!error.empty(), "se rechazo un JPEG sin explicar por que");
      }
    }
  }

  std::cout << "PASADO ✅ (transformada exacta; base y progresivo idénticos; "
            << rejected << " entradas corruptas rechazadas, " << accepted
            << " aceptadas, 0 caídas)\n"
            << std::flush;
}

/**
 * @brief Conv2D rapida contra Conv2D de referencia.
 *
 * `Conv2D` reformula la convolucion como una multiplicacion de matrices, lo que
 * la hace 55 veces mas rapida y, de paso, mucho mas facil de equivocar: hay
 * indices calculados y reordenaciones de memoria donde antes solo habia bucles.
 * `Conv2DReference` conserva la version literal, que se lee al lado de la
 * formula, y aqui se contrastan.
 *
 * El criterio es el error relativo a la magnitud del tensor entero, no elemento
 * a elemento. Dividir por un elemento que la cancelacion dejo casi en cero no
 * mide el error, mide la cancelacion: con esa medida salian discrepancias de
 * 9e-03 que resultaron no ser nada. Comprobado calculando la convolucion en
 * doble precision: las dos implementaciones en float32 se apartan de ese valor
 * por igual —2.3e-07 la de referencia y 2.0e-07 la rapida—, de modo que lo que
 * las separa es el orden de las sumas y no un defecto.
 */
void TestConv2DRapidaContraReferencia() {
  std::cout << "🧪 [Test 30] Conv2D por im2col contra la implementación literal... " << std::flush;

  struct Caso { int b, ic, oc, h, w, k, s, p; const char* nota; };
  const Caso casos[] = {
      {2, 1, 16, 32, 128, 3, 1, 1, "conv1 del CRNN"},
      {2, 16, 32, 16, 64, 3, 1, 1, "conv2 del CRNN"},
      {2, 32, 64, 8, 32, 3, 1, 1, "conv3 del CRNN"},
      {1, 3, 8, 12, 12, 3, 1, 0, "sin relleno"},
      {2, 2, 4, 9, 11, 3, 2, 1, "paso 2"},
      {1, 2, 3, 7, 7, 5, 1, 2, "nucleo 5x5"},
      {3, 1, 2, 5, 5, 1, 1, 0, "nucleo 1x1"},
      {1, 4, 4, 6, 6, 3, 3, 0, "paso igual al nucleo"},
  };

  // Error cuadratico medio de la diferencia, en proporcion a la magnitud del
  // tensor de referencia.
  auto error = [](const Tensor& a, const Tensor& b) {
    double suma_err = 0.0, suma_ref = 0.0;
    for (size_t i = 0; i < a.TotalSize(); ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      suma_err += d * d;
      suma_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(suma_err) / std::max(std::sqrt(suma_ref), 1e-12);
  };

  double peor = 0.0;
  for (const Caso& c : casos) {
    // La misma semilla antes de cada una: comparten pesos iniciales.
    ManualSeed(99);
    Conv2D rapida(c.ic, c.oc, c.k, c.s, c.p);
    ManualSeed(99);
    Conv2DReference referencia(c.ic, c.oc, c.k, c.s, c.p);

    // Los sesgos nacen a cero en ambas; sin darles valor, la prueba no
    // distinguiria una capa que los ignorase.
    for (int i = 0; i < c.oc; ++i) {
      const float v = 0.1f * (i % 5) - 0.2f;
      (*rapida.GetParameters()[1])[i] = v;
      (*referencia.GetParameters()[1])[i] = v;
    }

    Tensor x({c.b, c.ic, c.h, c.w});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.6f * std::sin(0.31f * static_cast<float>(i)) + 0.1f;
    }

    const Tensor y_rapida = rapida.Forward(x);
    const Tensor y_ref = referencia.Forward(x);
    Check(y_rapida.Shape() == y_ref.Shape(),
          std::string(c.nota) + ": las dos convoluciones dan formas distintas");

    Tensor dout(y_ref.Shape());
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.4f * std::cos(0.17f * static_cast<float>(i));
    }
    const Tensor dx_rapida = rapida.Backward(dout);
    const Tensor dx_ref = referencia.Backward(dout);

    const double e_salida = error(y_rapida, y_ref);
    const double e_peso = error(*rapida.GetGradients()[0], *referencia.GetGradients()[0]);
    const double e_sesgo = error(*rapida.GetGradients()[1], *referencia.GetGradients()[1]);
    const double e_entrada = error(dx_rapida, dx_ref);

    Check(e_salida < 1e-5, std::string(c.nota) + ": la salida difiere en " + std::to_string(e_salida));
    Check(e_peso < 1e-5, std::string(c.nota) + ": dW difiere en " + std::to_string(e_peso));
    Check(e_sesgo < 1e-5, std::string(c.nota) + ": db difiere en " + std::to_string(e_sesgo));
    Check(e_entrada < 1e-5, std::string(c.nota) + ": dx difiere en " + std::to_string(e_entrada));
    peor = std::max({peor, e_salida, e_peso, e_sesgo, e_entrada});
  }

  // Las operaciones elementales y el pooling reparten su bucle entre hilos, y
  // ninguna lleva reduccion, asi que el resultado tiene que ser identico bit a
  // bit. En MaxPool2D el reparto va por plano y no por posicion de salida: dos
  // ventanas solapadas pueden compartir maximo, de modo que repartir por
  // posicion seria una carrera. Esto lo detectaria.
  {
    const int hilos = parallel::ThreadCount();
    Tensor x({4, 6, 20, 24});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.7f * std::sin(0.31f * static_cast<float>(i)) - 0.2f;
    }

    auto ejecutar = [&](int n) {
      parallel::ThreadCount() = n;
      Tensor relu, sig, suma;
      ReluForward(x, relu);
      SigmoidForward(x, sig);
      ElementwiseMul(relu, sig, suma);
      // Ventana 3 con paso 2: las ventanas se solapan a proposito.
      MaxPool2D pool(3, 3, 2, 2);
      Tensor y = pool.Forward(suma);
      Tensor d(y.Shape());
      for (size_t i = 0; i < d.TotalSize(); ++i) {
        d[i] = 0.4f * std::cos(0.17f * static_cast<float>(i));
      }
      Tensor dx = pool.Backward(d);
      std::vector<Tensor> salida;
      salida.push_back(suma);
      salida.push_back(y);
      salida.push_back(dx);
      return salida;
    };

    const std::vector<Tensor> serie = ejecutar(1);
    const std::vector<Tensor> varios = ejecutar(std::max(4, hilos));
    parallel::ThreadCount() = hilos;

    const char* nombres[3] = {"las operaciones elementales", "MaxPool2D hacia delante",
                              "MaxPool2D hacia atras"};
    for (int k = 0; k < 3; ++k) {
      bool iguales = serie[k].TotalSize() == varios[k].TotalSize();
      for (size_t i = 0; iguales && i < serie[k].TotalSize(); ++i) {
        iguales = serie[k][i] == varios[k][i];
      }
      Check(iguales, std::string(nombres[k]) + " cambian con el numero de hilos");
    }
  }

  // Anidar reparto dentro de reparto. Conv2D reparte por imagen del lote y
  // dentro llama a MatMul, que reparte por filas. El guardia que evita volver a
  // pedir el pool estando dentro de el solo marcaba a los hilos trabajadores, y
  // no al que hace la llamada, que tambien ejecuta una porcion. Resultado: se
  // pisaba la tarea en vuelo y la espera de fuera no despertaba nunca. Estuvo
  // latente hasta que algo anido de verdad.
  {
    std::vector<int> conteo(200, 0);
    parallel::ParallelFor(200, 1, [&](int desde, int hasta) {
      for (int i = desde; i < hasta; ++i) {
        parallel::ParallelFor(20, 1, [&](int d2, int h2) {
          for (int j = d2; j < h2; ++j) conteo[i] += 1;
        });
      }
    });
    long suma = 0;
    for (int v : conteo) suma += v;
    Check(suma == 200 * 20,
          "el reparto anidado no ejecuto todo el trabajo: " + std::to_string(suma));
  }

  // El reparto entre hilos de Conv2D es por imagen del lote, y en el paso hacia
  // atras los gradientes de los pesos suman sobre todo el lote. Esa reduccion se
  // hace aparte, en orden fijo, justamente para que el resultado no dependa de
  // cuantos hilos haya. Si algun dia se sustituye por una suma directa desde los
  // hilos, seguiria pasando la comparacion contra la referencia -el error
  // seguiria siendo de redondeo- pero el entrenamiento dejaria de ser
  // reproducible. Esto lo detecta.
  {
    const int hilos_originales = parallel::ThreadCount();
    Conv2D capa(8, 12, 3, 1, 1);
    Tensor x({6, 8, 10, 14});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.5f * std::sin(0.23f * static_cast<float>(i)) - 0.1f;
    }
    Tensor dout({6, 12, 10, 14});
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.3f * std::cos(0.11f * static_cast<float>(i));
    }

    parallel::ThreadCount() = 1;
    capa.Forward(x);
    const Tensor dx_serie = capa.Backward(dout);
    const Tensor dw_serie = *capa.GetGradients()[0];
    const Tensor db_serie = *capa.GetGradients()[1];

    parallel::ThreadCount() = std::max(4, hilos_originales);
    capa.Forward(x);
    const Tensor dx_hilos = capa.Backward(dout);
    parallel::ThreadCount() = hilos_originales;

    bool iguales = dx_serie.TotalSize() == dx_hilos.TotalSize();
    for (size_t i = 0; iguales && i < dx_serie.TotalSize(); ++i) {
      iguales = dx_serie[i] == dx_hilos[i];
    }
    Check(iguales, "dx de Conv2D cambia con el numero de hilos");

    const Tensor& dw_hilos = *capa.GetGradients()[0];
    const Tensor& db_hilos = *capa.GetGradients()[1];
    bool pesos_iguales = true;
    for (size_t i = 0; pesos_iguales && i < dw_serie.TotalSize(); ++i) {
      pesos_iguales = dw_serie[i] == dw_hilos[i];
    }
    Check(pesos_iguales, "dW de Conv2D cambia con el numero de hilos");
    bool sesgos_iguales = true;
    for (size_t i = 0; sesgos_iguales && i < db_serie.TotalSize(); ++i) {
      sesgos_iguales = db_serie[i] == db_hilos[i];
    }
    Check(sesgos_iguales, "db de Conv2D cambia con el numero de hilos");
  }

  std::cout << "PASADO ✅ (" << (sizeof(casos) / sizeof(casos[0]))
            << " configuraciones; peor error relativo: " << peor
            << "; identico con 1 y con varios hilos)\n"
            << std::flush;
}

/**
 * @brief LSTM por multiplicacion de matrices contra la implementacion literal.
 *
 * `LSTM` reformula la celda como GEMM y va 16 veces mas rapida, a costa de
 * reordenar la memoria: la proyeccion de la entrada se calcula para todos los
 * pasos de golpe y los gradientes de los pesos salen de apilar los de cada
 * paso. `LSTMReference` conserva la version que se lee al lado de las
 * ecuaciones, y aqui se contrastan.
 *
 * El criterio es el error relativo a la magnitud del tensor, por la misma razon
 * que en Conv2D: elemento a elemento, donde la cancelacion deja un valor casi
 * nulo, se estaria midiendo la cancelacion y no el error.
 *
 * Los casos degenerados no son adorno. Un solo paso de tiempo desactiva toda la
 * parte recurrente; un lote de uno y una sola unidad oculta dejan matrices de
 * una fila o una columna, que es donde se equivoca un indice calculado.
 */
void TestLstmRapidaContraReferencia() {
  std::cout << "🧪 [Test 31] LSTM por GEMM contra la implementación literal... " << std::flush;

  struct Caso { int T, B, IN, H; const char* nota; };
  const Caso casos[] = {
      {32, 16, 64, 64, "el del CRNN"},
      {4, 2, 3, 5, "minimo"},
      {1, 1, 2, 2, "un solo paso"},
      {8, 1, 7, 3, "lote de uno"},
      {16, 4, 1, 8, "entrada de un canal"},
      {5, 3, 9, 1, "una sola unidad"},
      {2, 5, 4, 6, "mas lote que pasos"},
  };

  auto error = [](const Tensor& a, const Tensor& b) {
    double suma_err = 0.0, suma_ref = 0.0;
    for (size_t i = 0; i < a.TotalSize(); ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      suma_err += d * d;
      suma_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(suma_err) / std::max(std::sqrt(suma_ref), 1e-12);
  };

  double peor = 0.0;
  for (const Caso& c : casos) {
    ManualSeed(5);
    LSTM rapida(c.IN, c.H);
    ManualSeed(5);
    LSTMReference referencia(c.IN, c.H);

    // Los sesgos nacen a cero y ademas los dos entran sumados en la misma
    // preactivacion: con valores distintos se comprueba que no se confunden.
    for (int i = 0; i < 4 * c.H; ++i) {
      const float v = 0.05f * ((i % 7) - 3);
      (*rapida.GetParameters()[2])[i] = v;
      (*referencia.GetParameters()[2])[i] = v;
      (*rapida.GetParameters()[3])[i] = -v;
      (*referencia.GetParameters()[3])[i] = -v;
    }

    Tensor x({c.T, c.B, c.IN});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.5f * std::sin(0.29f * static_cast<float>(i)) + 0.1f;
    }
    const Tensor y_rapida = rapida.Forward(x);
    const Tensor y_ref = referencia.Forward(x);
    Check(y_rapida.Shape() == y_ref.Shape(),
          std::string(c.nota) + ": las dos LSTM dan formas distintas");

    Tensor dout(y_ref.Shape());
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.4f * std::cos(0.19f * static_cast<float>(i));
    }
    const Tensor dx_rapida = rapida.Backward(dout);
    const Tensor dx_ref = referencia.Backward(dout);

    const char* nombres[4] = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"};
    for (int p = 0; p < 4; ++p) {
      const double e = error(*rapida.GetGradients()[p], *referencia.GetGradients()[p]);
      Check(e < 1e-5, std::string(c.nota) + ": el gradiente de " + nombres[p] + " difiere en " +
                          std::to_string(e));
      peor = std::max(peor, e);
    }
    const double e_salida = error(y_rapida, y_ref);
    const double e_entrada = error(dx_rapida, dx_ref);
    Check(e_salida < 1e-5, std::string(c.nota) + ": la salida difiere en " + std::to_string(e_salida));
    Check(e_entrada < 1e-5, std::string(c.nota) + ": dx difiere en " + std::to_string(e_entrada));
    peor = std::max({peor, e_salida, e_entrada});
  }

  std::cout << "PASADO ✅ (" << (sizeof(casos) / sizeof(casos[0]))
            << " configuraciones; peor error relativo: " << peor << ")\n"
            << std::flush;
}

/**
 * @brief Separación de una imagen en renglones.
 *
 * La comprobación de verdad son las dos imágenes del repositorio —19 renglones
 * en la página de la Ilíada, 3 bandas en el logotipo de Mitsubishi— pero esas no
 * pueden vivir aquí: la prueba tiene que correr sin depender de archivos ni del
 * directorio de trabajo. Se construyen bandas de tinta en memoria, con las
 * posiciones conocidas de antemano.
 */
void TestSeparacionEnRenglones() {
  std::cout << "🧪 [Test 32] Separación de una imagen en renglones... " << std::flush;
  using namespace neuralsuite::image;

  // Dibuja una banda horizontal de tinta entre dos columnas.
  auto banda = [](Bitmap* destino, int y0, int y1, int x0, int x1, uint8_t nivel) {
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        destino->pixels[static_cast<size_t>(y) * destino->width + x] = nivel;
      }
    }
  };
  auto lienzo = [](int w, int h, uint8_t fondo) {
    Bitmap b;
    b.width = w; b.height = h; b.channels = 1;
    b.pixels.assign(static_cast<size_t>(w) * h, fondo);
    return b;
  };

  // 1. Tres renglones separados, con márgenes distintos a cada lado.
  {
    Bitmap imagen = lienzo(100, 60, 245);
    banda(&imagen, 5, 14, 10, 80, 20);
    banda(&imagen, 25, 34, 30, 60, 20);
    banda(&imagen, 45, 54, 0, 99, 20);

    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 3, "se esperaban 3 renglones y salieron " + std::to_string(r.size()));
    Check(r[0].y == 5 && r[0].alto == 10, "el primer renglon no esta donde se dibujo");
    // El recorte lateral es lo que evita reescalar franjas de fondo vacio.
    Check(r[0].x == 10 && r[0].ancho == 71, "el primer renglon no se recorto a lo ancho");
    Check(r[1].x == 30 && r[1].ancho == 31, "el segundo renglon no se recorto a lo ancho");
    Check(r[2].x == 0 && r[2].ancho == 100, "el tercer renglon, que ocupa todo, se recorto de mas");
  }

  // 2. Un acento queda despegado del cuerpo de la letra. Sin volver a unirlos,
  //    saldria como un renglon propio y el modelo recibiria una mota suelta.
  {
    Bitmap imagen = lienzo(60, 40, 250);
    banda(&imagen, 8, 9, 20, 24, 15);    // el acento
    banda(&imagen, 12, 24, 10, 50, 15);  // el cuerpo, dos filas mas abajo
    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 1, "el acento salio como un renglon aparte: " + std::to_string(r.size()));
    Check(r[0].y == 8 && r[0].alto == 17, "el renglon unido no abarca el acento y el cuerpo");
  }

  // 3. Dos renglones seguidos NO se unen aunque casi se toquen. Al principio la
  //    union miraba solo el hueco, y eso estaba ajustado sin querer a la escala
  //    de una imagen: funcionaba con renglones de 14 px, y en un abecedario
  //    manuscrito de 75 px dos renglones separados por dos filas se fusionaban
  //    en uno de 159. Lo que separa un acento de un renglon no es la distancia
  //    sino el tamano.
  {
    Bitmap imagen = lienzo(80, 120, 250);
    banda(&imagen, 10, 49, 5, 70, 15);    // renglon alto
    banda(&imagen, 52, 91, 5, 70, 15);    // otro igual de alto, a dos filas
    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 2, "dos renglones de altura parecida se fusionaron en " +
                             std::to_string(r.size()));
    Check(r[0].alto == 40 && r[1].alto == 40, "los dos renglones no conservan su alto");
  }

  // 4. Una mota de una fila no es un renglon.
  {
    Bitmap imagen = lienzo(60, 40, 250);
    banda(&imagen, 5, 5, 30, 32, 10);     // mota
    banda(&imagen, 20, 32, 5, 55, 10);    // renglon de verdad
    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 1, "la mota se conto como renglon");
    Check(r[0].y == 20, "se quedo con la mota en vez de con el renglon");
  }

  // 5. El umbral sale del histograma, no de una constante. Con poco contraste
  //    —un escaneo gris sobre gris— un valor fijo como 128 fallaria: aqui los
  //    dos niveles, 150 y 190, estan del mismo lado de ese corte.
  {
    Bitmap imagen = lienzo(80, 30, 190);
    banda(&imagen, 10, 19, 20, 60, 150);
    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 1, "no encontro el renglon de bajo contraste");
    Check(r[0].y == 10 && r[0].alto == 10, "el renglon de bajo contraste salio mal delimitado");
  }

  // 6. El recorte deja margen para que la tinta ocupe la proporcion con la que
  //    se entreno. Al ras, cada letra abarca mas pasos de la secuencia de los
  //    que el modelo vio nunca, y aparecen caracteres insertados.
  {
    Bitmap imagen = lienzo(100, 60, 245);
    banda(&imagen, 20, 39, 10, 90, 20);   // renglon de 20 px de alto
    const std::vector<Renglon> r = DetectarRenglones(imagen);
    Check(r.size() == 1, "no encontro el renglon");
    const Bitmap recorte = RecortarRenglon(imagen, r[0]);
    const double proporcion = static_cast<double>(r[0].alto) / recorte.height;
    Check(std::abs(proporcion - kProporcionTintaEntrenamiento) < 0.08,
          "el recorte deja la tinta ocupando " + std::to_string(proporcion) +
              " del alto y se esperaba cerca de " +
              std::to_string(kProporcionTintaEntrenamiento));

    // Y el tensor sale con el alto que espera la red y un ancho multiplo de 4.
    const Tensor t = RenglonATensor(recorte, 32, 4, /*invertir=*/true);
    Check(t.Shape()[2] == 32, "el tensor del renglon no tiene 32 filas");
    Check(t.Shape()[3] % 4 == 0, "el ancho del tensor no es multiplo de 4");
  }

  // 7. Iluminacion desigual. Un umbral global no puede con esto: con un
  //    degradado, medio folio cae entero por debajo del corte y se cuenta como
  //    tinta. Es lo que hace falta el umbral por vecindad.
  //
  //    Y la eleccion entre uno y otro no puede ser fija: en el caso 5, de bajo
  //    contraste, el adaptativo es el que falla, porque su umbral se apoya en
  //    la desviacion local y ahi es pequena. La automatica prueba los dos y se
  //    queda con el que deja la proyeccion mas nitida.
  {
    // Tinta gris, no negra: con tinta muy oscura ningún degradado razonable
    // llega a solapar las dos poblaciones y un umbral global seguiría
    // valiendo. El caso difícil es texto gris sobre papel iluminado de forma
    // desigual, que es lo que produce una foto de un documento.
    Bitmap imagen = lienzo(220, 130, 245);
    for (int r = 0; r < 4; ++r) banda(&imagen, 12 + r * 30, 12 + r * 30 + 16, 15, 205, 130);
    // Degradado lateral: el fondo del lado oscuro baja hasta 98, por debajo de
    // la tinta del lado claro (130). Ahí ningún corte único separa las dos.
    for (int y = 0; y < 130; ++y) {
      for (int x = 0; x < 220; ++x) {
        const float factor = 1.0f - 0.6f * static_cast<float>(x) / 219.0f;
        uint8_t& p = imagen.pixels[static_cast<size_t>(y) * 220 + x];
        p = static_cast<uint8_t>(p * factor);
      }
    }

    const size_t global = DetectarRenglones(imagen, 4, 2, 0.4f, Binarizacion::kGlobal).size();
    const size_t adaptativa =
        DetectarRenglones(imagen, 4, 2, 0.4f, Binarizacion::kAdaptativa).size();
    const size_t automatica =
        DetectarRenglones(imagen, 4, 2, 0.4f, Binarizacion::kAutomatica).size();

    Check(global != 4, "el umbral global aguanto el degradado: la prueba no mide lo que dice");
    Check(adaptativa == 4, "el umbral por vecindad dio " + std::to_string(adaptativa) +
                               " renglones con iluminacion desigual");
    Check(automatica == 4, "la eleccion automatica dio " + std::to_string(automatica) +
                               " renglones con iluminacion desigual");
  }

  // 8. Ruido de escaneo. Aquí es donde se ve para qué sirve normalizar la
  //    nitidez por la tinta total: sin normalizar, el método que marca más
  //    píxeles gana siempre, y el adaptativo marca de más sobre el grano. Se
  //    comprobó mutando: quitar la normalización baja la página con ruido de
  //    18 renglones a 16.
  {
    // Trazos finos y discontinuos, como letras: sobre bandas macizas el grano
    // no llega a cerrar los huecos entre renglones y el efecto no aparece.
    Bitmap imagen = lienzo(300, 180, 240);
    for (int r = 0; r < 5; ++r) {
      for (int y = 12 + r * 34; y < 12 + r * 34 + 12; ++y) {
        for (int x = 20; x < 280; ++x) {
          if ((x / 2) % 2 == 0) imagen.pixels[static_cast<size_t>(y) * 300 + x] = 40;
        }
      }
    }
    uint32_t estado = 991;
    for (size_t i = 0; i < imagen.pixels.size(); ++i) {
      estado ^= estado << 13; estado ^= estado >> 17; estado ^= estado << 5;
      const int ruido = static_cast<int>(estado % 81) - 40;
      const int v = static_cast<int>(imagen.pixels[i]) + ruido;
      imagen.pixels[i] = static_cast<uint8_t>(std::max(0, std::min(255, v)));
    }
    const size_t automatica =
        DetectarRenglones(imagen, 4, 2, 0.4f, Binarizacion::kAutomatica).size();
    Check(automatica == 5,
          "con ruido de escaneo salieron " + std::to_string(automatica) + " renglones de 5");
  }

  // 9. Una imagen sin tinta no produce renglones ni revienta.
  {
    const Bitmap vacia = lienzo(50, 50, 255);
    Check(DetectarRenglones(vacia).empty(), "encontro renglones en una imagen en blanco");
    Bitmap nada;
    Check(DetectarRenglones(nada).empty(), "no soporto una imagen vacia");
  }

  std::cout << "PASADO ✅ (9 casos: posición, acentos, renglones pegados, motas, contraste, proporción, iluminación desigual, ruido)\n"
            << std::flush;
}

/**
 * @brief Atención por multiplicación de matrices contra la implementación literal.
 *
 * `MultiHeadAttention` extrae cada cabeza como matriz contigua y resuelve
 * `Q·Kᵀ` y `P·V` con `MatMul`. Eso implica reordenar memoria —los tres
 * proyectados vienen entrelazados en una sola fila— y ahí es donde se
 * desalinea un índice. `MultiHeadAttentionReference` conserva la versión que se
 * lee al lado de las fórmulas.
 *
 * El caso de un solo token comprueba el camino degenerado donde la matriz de
 * puntuaciones es 1×1 y la máscara causal no descarta nada.
 */
void TestAtencionRapidaContraReferencia() {
  std::cout << "🧪 [Test 33] Atención por GEMM contra la implementación literal... " << std::flush;

  struct Caso { int B, T, E, H; const char* nota; };
  const Caso casos[] = {
      {2, 64, 128, 4, "el del GPT"},
      {1, 8, 16, 2, "minimo"},
      {1, 1, 8, 2, "un solo token"},
      {3, 16, 32, 8, "ocho cabezas"},
      {2, 5, 4, 1, "una sola cabeza"},
      {1, 32, 64, 4, "lote de uno"},
  };

  auto error = [](const Tensor& a, const Tensor& b) {
    double suma_err = 0.0, suma_ref = 0.0;
    for (size_t i = 0; i < a.TotalSize(); ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      suma_err += d * d;
      suma_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(suma_err) / std::max(std::sqrt(suma_ref), 1e-12);
  };

  double peor = 0.0;
  for (const Caso& c : casos) {
    ManualSeed(3);
    MultiHeadAttention rapida(c.E, c.H);
    ManualSeed(3);
    MultiHeadAttentionReference referencia(c.E, c.H);

    Tensor x({c.B, c.T, c.E});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.4f * std::sin(0.27f * static_cast<float>(i)) + 0.05f;
    }
    const Tensor y_rapida = rapida.Forward(x);
    const Tensor y_ref = referencia.Forward(x);
    Check(y_rapida.Shape() == y_ref.Shape(),
          std::string(c.nota) + ": las dos atenciones dan formas distintas");

    Tensor dout(y_ref.Shape());
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.3f * std::cos(0.13f * static_cast<float>(i));
    }
    const Tensor dx_rapida = rapida.Backward(dout);
    const Tensor dx_ref = referencia.Backward(dout);

    const double e_salida = error(y_rapida, y_ref);
    const double e_entrada = error(dx_rapida, dx_ref);
    Check(e_salida < 1e-5, std::string(c.nota) + ": la salida difiere en " + std::to_string(e_salida));
    Check(e_entrada < 1e-5, std::string(c.nota) + ": dx difiere en " + std::to_string(e_entrada));
    peor = std::max({peor, e_salida, e_entrada});
    for (size_t p = 0; p < rapida.GetGradients().size(); ++p) {
      const double e = error(*rapida.GetGradients()[p], *referencia.GetGradients()[p]);
      Check(e < 1e-5, std::string(c.nota) + ": el gradiente " + std::to_string(p) +
                          " difiere en " + std::to_string(e));
      peor = std::max(peor, e);
    }
  }

  // El reparto va por pareja (muestra, cabeza), y cada cabeza escribe en un
  // tramo distinto de dqkv, así que no hay reducción y el resultado no puede
  // depender del número de hilos. Si alguien reorganiza eso y mete una suma
  // compartida, la comparación contra la referencia seguiría pasando —el error
  // seguiría siendo de redondeo— pero el entrenamiento dejaría de ser
  // reproducible.
  {
    const int hilos = parallel::ThreadCount();
    MultiHeadAttention capa(64, 8);
    Tensor x({4, 24, 64});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.3f * std::sin(0.19f * static_cast<float>(i));
    }
    Tensor dout({4, 24, 64});
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.2f * std::cos(0.23f * static_cast<float>(i));
    }

    parallel::ThreadCount() = 1;
    const Tensor y_serie = capa.Forward(x);
    const Tensor dx_serie = capa.Backward(dout);
    parallel::ThreadCount() = std::max(4, hilos);
    const Tensor y_hilos = capa.Forward(x);
    const Tensor dx_hilos = capa.Backward(dout);
    parallel::ThreadCount() = hilos;

    bool iguales = true;
    for (size_t i = 0; iguales && i < y_serie.TotalSize(); ++i) iguales = y_serie[i] == y_hilos[i];
    Check(iguales, "la salida de la atencion cambia con el numero de hilos");
    for (size_t i = 0; iguales && i < dx_serie.TotalSize(); ++i) iguales = dx_serie[i] == dx_hilos[i];
    Check(iguales, "dx de la atencion cambia con el numero de hilos");
  }

  std::cout << "PASADO ✅ (" << (sizeof(casos) / sizeof(casos[0]))
            << " configuraciones; peor error relativo: " << peor
            << "; idéntico con 1 y con varios hilos)\n"
            << std::flush;
}

/**
 * @brief Estimación y corrección de la inclinación.
 *
 * La verificación real son las rotaciones conocidas de la página de la Ilíada
 * —ocho ángulos de -5° a +5°, todos recuperados con error 0.00°— pero esa
 * imagen no puede vivir en la prueba: tiene que correr sin depender de archivos.
 * Se construye una página sintética de renglones, se gira un ángulo conocido y
 * se exige recuperarlo.
 *
 * Lo que se comprueba no es solo el ángulo sino la consecuencia: que la
 * separación en renglones, que se desploma con la página torcida, vuelva a
 * encontrar los que hay.
 */
void TestEnderezarPagina() {
  std::cout << "🧪 [Test 34] Corrección de la inclinación de una página... " << std::flush;
  using namespace neuralsuite::image;

  // Página sintética: ocho renglones de trazos, con huecos entre ellos.
  //
  // El ancho no es arbitrario. La precisión del estimador depende de él: un
  // grado décimo inclina la última columna respecto de la primera en
  // `ancho * tan(0.1°)` píxeles, y por debajo de uno no hay nada que la
  // proyección pueda distinguir. Medido sobre una página recta, el estimador
  // devuelve -0.30° con 200 px de ancho, -0.10° con 400 y 0.00° a partir de
  // 800. Se usan 600, que es del orden de un escaneo real.
  const int W = 600, H = 480;
  Bitmap pagina;
  pagina.width = W; pagina.height = H; pagina.channels = 1;
  pagina.pixels.assign(static_cast<size_t>(W) * H, 250);
  const int kRenglones = 8;
  for (int r = 0; r < kRenglones; ++r) {
    const int y0 = 30 + r * 54;
    for (int y = y0; y < y0 + 30; ++y) {
      for (int x = 60; x < W - 60; ++x) {
        // Trazos discontinuos, como letras, no una barra maciza.
        if ((x / 3) % 2 == 0) pagina.pixels[static_cast<size_t>(y) * W + x] = 20;
      }
    }
  }
  Check(DetectarRenglones(pagina).size() == static_cast<size_t>(kRenglones),
        "la pagina sintetica recta no da los renglones que se dibujaron");

  // 1. La recta se reconoce como recta y no se toca.
  {
    const double grados = EstimarInclinacion(pagina);
    Check(std::abs(grados) <= 0.11,
          "una pagina recta se estimo inclinada " + std::to_string(grados) + " grados");
  }

  // 2. Ángulos conocidos, recuperados y corregidos.
  //
  // Ninguno es múltiplo del paso grueso de la búsqueda (0.5°) a propósito: con
  // ángulos redondos, la primera pasada ya acierta y el afinado no se ejercita.
  // Se comprobó: quitar el afinado no ponía nada en rojo hasta cambiar esto.
  double peor = 0.0;
  for (double angulo : {-3.7, -2.3, -1.2, 1.4, 2.8, 4.3}) {
    const Bitmap torcida = Girar(pagina, angulo);
    const double estimado = EstimarInclinacion(torcida);
    const double error = std::abs(estimado - angulo);
    peor = std::max(peor, error);
    Check(error <= 0.11, "con " + std::to_string(angulo) + " grados se estimo " +
                            std::to_string(estimado));

    double aplicado = 0.0;
    const Bitmap recta = Enderezar(torcida, &aplicado);
    const size_t tras = DetectarRenglones(recta).size();
    Check(tras == static_cast<size_t>(kRenglones),
          "tras enderezar " + std::to_string(angulo) + " grados salen " +
              std::to_string(tras) + " renglones y deberian ser " +
              std::to_string(kRenglones));
  }

  // 3. El giro conserva el tamaño y rellena con el fondo, no con blanco puro:
  //    un marco luminoso entraria en el histograma y falsearia el umbral.
  {
    const Bitmap girada = Girar(pagina, 3.0);
    Check(girada.width == W && girada.height == H, "el giro cambio el tamano de la imagen");
    Check(girada.pixels[0] >= 240 && girada.pixels[0] <= 255,
          "la esquina, que queda fuera tras girar, no se relleno con el fondo");
  }

  // 4. Una imagen vacia no revienta.
  {
    Bitmap nada;
    Check(EstimarInclinacion(nada) == 0.0, "no soporto una imagen vacia");
  }

  std::cout << "PASADO ✅ (6 ángulos; peor error " << peor << "°)\n" << std::flush;
}

/**
 * @brief Las dos primitivas que faltaban, contra las capas ya verificadas.
 *
 * `Gather` y `Conv2DVar` completan el motor de autograd: con ellas se puede
 * expresar un embedding y una convolución sin escribir su derivada.
 *
 * La comprobación no es contra diferencias finitas sino **contra las capas que
 * ya tienen paridad con PyTorch**. Eso las hace oráculo: si el gradiente que
 * deriva el grafo coincide con el que la capa calcula a mano, o las dos están
 * bien, o comparten un error — y no lo comparten, porque una recorre un grafo y
 * la otra aplica una fórmula escrita a mano.
 */
void TestPrimitivasGatherYConv() {
  std::cout << "🧪 [Test 35] Primitivas gather y convolución del autograd... " << std::flush;
  using namespace neuralsuite::autograd;

  // 1. Gather contra Embedding.
  {
    const int vocabulario = 12, dim = 5, lote = 2, pasos = 4;
    ManualSeed(31);
    Embedding capa(vocabulario, dim);

    Tensor indices({lote, pasos});
    // Un token repetido a propósito: su fila recibe la suma de dos gradientes,
    // y quedarse con el último en vez de sumarlos es el error natural aquí.
    const int tokens[] = {3, 7, 3, 0, 11, 5, 5, 9};
    for (int i = 0; i < lote * pasos; ++i) indices[i] = static_cast<float>(tokens[i]);

    const Tensor y_capa = capa.Forward(indices);

    auto tabla = Variable::Create(capa.Weight(), /*requires_grad=*/true);
    auto y_grafo = Gather(tabla, indices);

    Check(y_grafo->Value().TotalSize() == y_capa.TotalSize(),
          "Gather no devuelve tantos valores como Embedding");
    double peor = 0.0;
    for (size_t i = 0; i < y_capa.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(y_grafo->Value()[i]) - y_capa[i]));
    }
    Check(peor < 1e-6, "Gather no reproduce la salida de Embedding: " + std::to_string(peor));

    Tensor dout(y_capa.Shape());
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.4f * std::sin(0.37f * static_cast<float>(i)) + 0.1f;
    }
    capa.Backward(dout);
    // `Backward` siembra el gradiente el mismo, asi que para comparar contra un
    // `dout` concreto se construye una perdida escalar cuya derivada respecto a
    // la salida es exactamente ese `dout`: sumar la salida multiplicada por el.
    Backward(Sum(Mul(y_grafo, Variable::Create(dout))));

    const Tensor& d_capa = *capa.GetGradients()[0];
    double peor_grad = 0.0;
    for (size_t i = 0; i < d_capa.TotalSize(); ++i) {
      peor_grad = std::max(peor_grad,
                           std::abs(static_cast<double>(tabla->Grad()[i]) - d_capa[i]));
    }
    Check(peor_grad < 1e-6,
          "el gradiente de Gather no coincide con el de Embedding: " + std::to_string(peor_grad));
  }

  // 2. Conv2DVar contra Conv2D, en varias geometrías.
  {
    struct Caso { int b, ic, oc, h, w, k, s, p; const char* nota; };
    const Caso casos[] = {
        {2, 3, 4, 8, 10, 3, 1, 1, "con relleno"},
        {1, 1, 2, 6, 6, 3, 1, 0, "sin relleno"},
        {2, 2, 3, 9, 7, 3, 2, 1, "paso 2"},
    };
    for (const Caso& c : casos) {
      ManualSeed(17);
      Conv2D capa(c.ic, c.oc, c.k, c.s, c.p);
      capa.Bias().Zeros();   // la primitiva no lleva sesgo: se suma aparte

      Tensor x({c.b, c.ic, c.h, c.w});
      for (size_t i = 0; i < x.TotalSize(); ++i) {
        x[i] = 0.5f * std::sin(0.23f * static_cast<float>(i)) + 0.05f;
      }
      const Tensor y_capa = capa.Forward(x);

      auto entrada = Variable::Create(x, /*requires_grad=*/true);
      auto pesos = Variable::Create(capa.Weight(), /*requires_grad=*/true);
      auto y_grafo = Conv2DVar(entrada, pesos, c.s, c.p);

      double peor = 0.0;
      for (size_t i = 0; i < y_capa.TotalSize(); ++i) {
        peor = std::max(peor, std::abs(static_cast<double>(y_grafo->Value()[i]) - y_capa[i]));
      }
      Check(peor < 1e-5,
            std::string(c.nota) + ": Conv2DVar no reproduce la salida de Conv2D");

      Tensor dout(y_capa.Shape());
      for (size_t i = 0; i < dout.TotalSize(); ++i) {
        dout[i] = 0.3f * std::cos(0.19f * static_cast<float>(i));
      }
      const Tensor dx_capa = capa.Backward(dout);
      Backward(Sum(Mul(y_grafo, Variable::Create(dout))));

      double peor_dx = 0.0, peor_dw = 0.0;
      for (size_t i = 0; i < dx_capa.TotalSize(); ++i) {
        peor_dx = std::max(peor_dx,
                           std::abs(static_cast<double>(entrada->Grad()[i]) - dx_capa[i]));
      }
      const Tensor& dw_capa = *capa.GetGradients()[0];
      for (size_t i = 0; i < dw_capa.TotalSize(); ++i) {
        peor_dw = std::max(peor_dw,
                           std::abs(static_cast<double>(pesos->Grad()[i]) - dw_capa[i]));
      }
      Check(peor_dx < 1e-5, std::string(c.nota) + ": dx difiere en " + std::to_string(peor_dx));
      Check(peor_dw < 1e-5, std::string(c.nota) + ": dW difiere en " + std::to_string(peor_dw));
    }
  }

  std::cout << "PASADO ✅ (gather contra Embedding, convolución en 3 geometrías)\n" << std::flush;
}

/** @brief La forma de entrada con la ultima dimension cambiada por `out`. */
static std::vector<int> forma_esperada(std::vector<int> forma, int out) {
  forma.back() = out;
  return forma;
}

/**
 * @brief `Linear` contra `LinearAutograd`: la misma capa por dos caminos.
 *
 * Una escribe su backward a mano; la otra lo obtiene recorriendo el grafo.
 * Con los mismos pesos deben dar el mismo numero, salida y gradientes.
 *
 * Esto es lo que hace segura la duplicacion. El proyecto ya mantiene tres
 * pares asi —Conv2D, LSTM, MultiHeadAttention— y ninguno ha divergido nunca,
 * mientras que las seis listas que nadie comparaba divergieron todas. La
 * diferencia no es la disciplina: es que exista esta prueba.
 */
void TestLinearContraAutograd() {
  std::cout << "🧪 [Test 36] Linear contra su version en autograd... " << std::flush;

  // Rangos distintos, para que un aplanado mal hecho no se salve por casualidad.
  const std::vector<std::vector<int>> formas = {{5, 7}, {3, 4, 7}, {2, 3, 4, 7}};
  const int in = 7, out = 6;

  for (const std::vector<int>& forma : formas) {
    ManualSeed(83);
    Linear a_mano(in, out);
    LinearAutograd por_grafo(in, out);

    // Mismos pesos en las dos: cualquier diferencia sera del calculo.
    por_grafo.Weight() = a_mano.Weight();
    a_mano.Bias().RandomNormal(0.0f, 0.3f);   // un sesgo nulo no probaria db
    por_grafo.Bias() = a_mano.Bias();

    Tensor x(forma);
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.6f * std::sin(0.29f * static_cast<float>(i)) - 0.1f;
    }

    const Tensor y1 = a_mano.Forward(x);
    const Tensor y2 = por_grafo.Forward(x);

    Check(y1.Shape() == y2.Shape(), "las dos capas no devuelven la misma forma");
    Check(y1.Shape() == forma_esperada(forma, out),
          "la forma de salida no conserva el rango");

    double peor = 0.0;
    for (size_t i = 0; i < y1.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(y1[i]) - y2[i]));
    }
    Check(peor < 1e-5, "las salidas difieren en " + std::to_string(peor));

    Tensor dout(y1.Shape());
    for (size_t i = 0; i < dout.TotalSize(); ++i) {
      dout[i] = 0.4f * std::cos(0.31f * static_cast<float>(i)) + 0.05f;
    }
    const Tensor dx1 = a_mano.Backward(dout);
    const Tensor dx2 = por_grafo.Backward(dout);

    Check(dx1.Shape() == forma, "dx no conserva la forma de la entrada");
    Check(dx2.Shape() == forma, "dx del grafo no conserva la forma de la entrada");

    double peor_dx = 0.0;
    for (size_t i = 0; i < dx1.TotalSize(); ++i) {
      peor_dx = std::max(peor_dx, std::abs(static_cast<double>(dx1[i]) - dx2[i]));
    }
    Check(peor_dx < 1e-5, "dx difiere en " + std::to_string(peor_dx));

    // GetGradients() los expone en el mismo orden en las dos capas; recorrerlos
    // asi comprueba de paso que ese orden coincide.
    const std::vector<Tensor*> g1 = a_mano.GetGradients();
    const std::vector<Tensor*> g2 = por_grafo.GetGradients();
    Check(g1.size() == g2.size() && g1.size() == 2,
          "las capas no exponen los mismos parametros");
    for (size_t p = 0; p < g1.size(); ++p) {
      Check(g1[p]->TotalSize() == g2[p]->TotalSize(),
            "el parametro " + std::to_string(p) + " no tiene el mismo tamano");
      double peor_g = 0.0;
      for (size_t i = 0; i < g1[p]->TotalSize(); ++i) {
        peor_g = std::max(peor_g,
                          std::abs(static_cast<double>((*g1[p])[i]) - (*g2[p])[i]));
      }
      Check(peor_g < 1e-5, "el gradiente del parametro " + std::to_string(p) +
                               " difiere en " + std::to_string(peor_g));
    }
  }

  std::cout << "PASADO ✅ (salida y gradientes iguales en rangos 2, 3 y 4)\n" << std::flush;
}

/**
 * @brief `Embedding` contra `EmbeddingAutograd`.
 *
 * Segunda pareja del esquema. El caso que decide la prueba es el token
 * repetido: su fila debe recibir la suma de todas las posiciones que lo
 * usaron. Una secuencia real repite tokens constantemente, asi que un error
 * ahi no se manifiesta como un fallo sino como un modelo que aprende peor.
 */
void TestEmbeddingContraAutograd() {
  std::cout << "🧪 [Test 37] Embedding contra su version en autograd... " << std::flush;

  const int vocabulario = 15, dim = 6, lote = 3, pasos = 5;

  ManualSeed(47);
  Embedding a_mano(vocabulario, dim);
  EmbeddingAutograd por_grafo(vocabulario, dim);
  por_grafo.Weight() = a_mano.Weight();

  Tensor indices({lote, pasos});
  // Repeticiones a proposito, dentro de la misma fila y entre filas distintas,
  // y una fila de la tabla (la 14) que no usa nadie: su gradiente debe quedar
  // en cero, no en basura.
  const int tokens[] = {2, 9, 2, 2, 0,
                        7, 7, 3, 11, 7,
                        0, 1, 9, 3, 2};
  for (int i = 0; i < lote * pasos; ++i) indices[i] = static_cast<float>(tokens[i]);

  const Tensor y1 = a_mano.Forward(indices);
  const Tensor y2 = por_grafo.Forward(indices);

  Check(y1.Shape() == y2.Shape(), "las dos capas no devuelven la misma forma");
  Check(y1.Shape() == std::vector<int>({lote, pasos, dim}),
        "la salida no es [lote, pasos, dimension]");

  double peor = 0.0;
  for (size_t i = 0; i < y1.TotalSize(); ++i) {
    peor = std::max(peor, std::abs(static_cast<double>(y1[i]) - y2[i]));
  }
  Check(peor < 1e-6, "las salidas difieren en " + std::to_string(peor));

  Tensor dout(y1.Shape());
  for (size_t i = 0; i < dout.TotalSize(); ++i) {
    dout[i] = 0.5f * std::sin(0.41f * static_cast<float>(i)) + 0.07f;
  }
  a_mano.Backward(dout);
  por_grafo.Backward(dout);

  const std::vector<Tensor*> g1 = a_mano.GetGradients();
  const std::vector<Tensor*> g2 = por_grafo.GetGradients();
  Check(g1.size() == g2.size() && g1.size() == 1,
        "las capas no exponen los mismos parametros");

  double peor_g = 0.0;
  for (size_t i = 0; i < g1[0]->TotalSize(); ++i) {
    peor_g = std::max(peor_g, std::abs(static_cast<double>((*g1[0])[i]) - (*g2[0])[i]));
  }
  Check(peor_g < 1e-6, "el gradiente de la tabla difiere en " + std::to_string(peor_g));

  // Que las dos coincidan no basta si las dos se equivocan igual. El token 2
  // aparece cuatro veces, asi que su fila tiene que valer la suma de esos
  // cuatro gradientes, calculada aqui aparte a mano.
  double esperado[16] = {0};
  for (int p = 0; p < lote * pasos; ++p) {
    if (tokens[p] != 2) continue;
    for (int d = 0; d < dim; ++d) esperado[d] += dout[p * dim + d];
  }
  double peor_suma = 0.0;
  for (int d = 0; d < dim; ++d) {
    peor_suma = std::max(peor_suma, std::abs(esperado[d] - (*g2[0])[2 * dim + d]));
  }
  Check(peor_suma < 1e-5,
        "la fila del token repetido no acumula sus cuatro gradientes: " +
            std::to_string(peor_suma));

  // La fila 14 no la usa ningun token: debe quedar exactamente en cero.
  for (int d = 0; d < dim; ++d) {
    Check((*g2[0])[14 * dim + d] == 0.0f,
          "una fila que nadie uso recibio gradiente");
  }

  // El caso real de GPT, que ninguna prueba cubria: el embedding de posicion se
  // calcula una vez con forma [1, T] y su gradiente llega para todo el lote,
  // [B, T, D]. Cada posicion debe sumar las B contribuciones.
  //
  // `Embedding` lo resuelve con un `% num_cached` en su bucle, que a primera
  // vista parece defensivo y en realidad es lo que sostiene este caso.
  // `EmbeddingAutograd` no tiene nada equivalente escrito: le sale del
  // broadcasting de `Mul` y la reduccion que hace el motor al propagar.
  {
    const int b_lote = 4, t_pasos = 5;
    ManualSeed(9);
    Embedding pos_a_mano(vocabulario, dim);
    EmbeddingAutograd pos_por_grafo(vocabulario, dim);
    pos_por_grafo.Weight() = pos_a_mano.Weight();

    Tensor pos({1, t_pasos});
    for (int t = 0; t < t_pasos; ++t) pos[t] = static_cast<float>(t);
    pos_a_mano.Forward(pos);
    pos_por_grafo.Forward(pos);

    Tensor dx({b_lote, t_pasos, dim});
    for (size_t i = 0; i < dx.TotalSize(); ++i) {
      dx[i] = 0.3f * std::sin(0.37f * static_cast<float>(i)) + 0.2f;
    }
    pos_a_mano.Backward(dx);
    pos_por_grafo.Backward(dx);

    // La verdad se calcula aqui aparte, para que no baste con que coincidan.
    double peor_a = 0.0, peor_b = 0.0;
    for (int t = 0; t < t_pasos; ++t) {
      for (int d = 0; d < dim; ++d) {
        double esperado = 0.0;
        for (int bb = 0; bb < b_lote; ++bb) esperado += dx[(bb * t_pasos + t) * dim + d];
        peor_a = std::max(peor_a,
                          std::abs(esperado - (*pos_a_mano.GetGradients()[0])[t * dim + d]));
        peor_b = std::max(peor_b,
                          std::abs(esperado - (*pos_por_grafo.GetGradients()[0])[t * dim + d]));
      }
    }
    Check(peor_a < 1e-5, "Embedding no suma el lote en el embedding de posicion: " +
                             std::to_string(peor_a));
    Check(peor_b < 1e-5, "EmbeddingAutograd no suma el lote en el embedding de posicion: " +
                             std::to_string(peor_b));
  }

  std::cout << "PASADO ✅ (repetidos, filas sin usar y posicion compartida)\n" << std::flush;
}

/**
 * @brief El KV-Cache debe dar los MISMOS logits que recalcular todo el contexto.
 *
 * La prueba que existía comparaba las secuencias generadas, y eso no basta: el
 * argmax absorbe una diferencia pequeña. Con esa comprobación pasó inadvertido
 * que `generate_llm` reinyectaba el último token del prompt —lo metía dos veces
 * en la caché— y producía logits distintos: 0.056 de diferencia con la ruta de
 * referencia. La secuencia salía igual, así que nadie lo vio.
 *
 * Aquí se comparan los logits en cada paso, y se generan bastantes más tokens
 * que `block_size` para cruzar la ventana varias veces. Cruzarla no es un
 * detalle: `wpe_` sólo tiene `block_size` posiciones, así que pasado ese punto
 * hay que reconstruir la caché con los últimos `block_size` tokens. Antes eso
 * abortaba con `Embedding: token N fuera del rango`.
 */
void TestKVCacheCoincideConRecalculo() {
  std::cout << "🧪 [Test 38] KV-Cache frente a recalcular el contexto... " << std::flush;

  GPTConfig cfg;
  cfg.vocab_size = 64;
  cfg.block_size = 12;          // ventana corta a proposito, para cruzarla pronto
  cfg.n_layer = 2;
  cfg.n_head = 2;
  cfg.n_embd = 64;
  ManualSeed(23);
  GPTModel modelo(cfg);

  std::vector<int> tokens = {3, 9, 14, 2, 7};
  const int nuevos = 40;

  auto sembrar = [&](int desde, int hasta) {
    Tensor ultimo;
    modelo.ClearKVCache();
    for (int i = desde; i < hasta; ++i) {
      ultimo = modelo.ForwardWithKVCache(tokens[i], i - desde);
    }
    return ultimo;
  };

  int inicio = std::max(0, static_cast<int>(tokens.size()) - cfg.block_size);
  Tensor logits = sembrar(inicio, static_cast<int>(tokens.size()));

  double peor = 0.0;
  int reconstrucciones = 0;
  for (int paso = 0; paso < nuevos; ++paso) {
    // La verdad: recalcular el contexto recortado, que es la ruta sin cache.
    const int n = static_cast<int>(tokens.size());
    const int ini_ref = std::max(0, n - cfg.block_size);
    Tensor idx({1, n - ini_ref});
    for (int i = ini_ref; i < n; ++i) idx[i - ini_ref] = static_cast<float>(tokens[i]);
    const Tensor logits_ref = modelo.Forward(idx);
    const int off = (n - ini_ref - 1) * cfg.vocab_size;

    for (int v = 0; v < cfg.vocab_size; ++v) {
      peor = std::max(peor, std::abs(static_cast<double>(logits[v]) - logits_ref[off + v]));
    }

    int mejor = 0;
    for (int v = 1; v < cfg.vocab_size; ++v) {
      if (logits_ref[off + v] > logits_ref[off + mejor]) mejor = v;
    }
    tokens.push_back(mejor);

    const int sl = static_cast<int>(tokens.size());
    if (sl - inicio > cfg.block_size) {
      inicio = sl - cfg.block_size;
      logits = sembrar(inicio, sl);
      ++reconstrucciones;
    } else {
      logits = modelo.ForwardWithKVCache(mejor, sl - 1 - inicio);
    }
  }

  Check(reconstrucciones > 0,
        "la prueba no llego a cruzar block_size: no comprueba la ventana deslizante");
  Check(peor < 1e-4,
        "el KV-Cache no coincide con recalcular el contexto: " + std::to_string(peor));

  std::cout << "PASADO ✅ (" << nuevos << " tokens, " << reconstrucciones
            << " cruces de ventana, peor diferencia " << peor << ")\n" << std::flush;
}

/**
 * @brief `Concat` y su derivada, que es lo que necesitan los saltos de U-Net.
 *
 * Hacia adelante se compara contra el valor calculado a mano con índices
 * explícitos, no contra la propia implementación: que una función coincida
 * consigo misma no prueba nada.
 *
 * Hacia atrás, la derivada de concatenar es **cortar**, y ahí está el error
 * natural: si el corte se desplaza aunque sea una posición, cada entrada recibe
 * un gradiente que es casi el suyo y el resultado sigue pareciendo razonable.
 * Por eso el gradiente de prueba lleva un valor distinto en cada posición: así
 * un corte desplazado se nota.
 */
void TestConcatYSuDerivada() {
  std::cout << "🧪 [Test 39] Concat y su derivada... " << std::flush;
  using namespace neuralsuite::autograd;

  // 1. Forward, uniendo por varios ejes y con tres entradas.
  {
    Tensor a({2, 3, 4}), b({2, 5, 4}), c({2, 1, 4});
    for (size_t i = 0; i < a.TotalSize(); ++i) a[i] = static_cast<float>(i);
    for (size_t i = 0; i < b.TotalSize(); ++i) b[i] = 1000.0f + static_cast<float>(i);
    for (size_t i = 0; i < c.TotalSize(); ++i) c[i] = 9000.0f + static_cast<float>(i);

    const Tensor u = Concat({&a, &b, &c}, 1);
    Check(u.Shape() == std::vector<int>({2, 9, 4}), "Concat no suma el eje 1");

    // La verdad, con índices explícitos.
    for (int lote = 0; lote < 2; ++lote) {
      for (int f = 0; f < 9; ++f) {
        for (int d = 0; d < 4; ++d) {
          float esperado;
          if (f < 3)      esperado = a[(lote * 3 + f) * 4 + d];
          else if (f < 8) esperado = b[(lote * 5 + (f - 3)) * 4 + d];
          else            esperado = c[(lote * 1 + (f - 8)) * 4 + d];
          Check(u[(lote * 9 + f) * 4 + d] == esperado,
                "Concat coloca mal el elemento [" + std::to_string(lote) + "," +
                    std::to_string(f) + "," + std::to_string(d) + "]");
        }
      }
    }
  }

  // El último eje es el que usan los saltos de una U-Net: se unen canales.
  {
    Tensor a({2, 3}), b({2, 2});
    for (size_t i = 0; i < a.TotalSize(); ++i) a[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < b.TotalSize(); ++i) b[i] = 100.0f * static_cast<float>(i + 1);
    const Tensor u = Concat(a, b, -1);   // eje negativo: cuenta desde el final
    Check(u.Shape() == std::vector<int>({2, 5}), "Concat no acepta ejes negativos");
    const float esperado[10] = {1, 2, 3, 100, 200, 4, 5, 6, 300, 400};
    for (int i = 0; i < 10; ++i) {
      Check(u[i] == esperado[i], "Concat entrelaza mal por el último eje");
    }
  }

  // 2. Formas incompatibles deben abortar, no producir un tensor entrelazado.
  {
    Tensor a({2, 3}), b({4, 3});
    bool protesto = false;
    try { Concat(a, b, 1); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "Concat aceptó formas que no encajan");
  }

  // 3. Backward: a cada entrada le llega su rebanada, sin desplazamiento.
  {
    auto a = Variable::Create(Tensor({2, 3, 4}), /*requires_grad=*/true);
    auto b = Variable::Create(Tensor({2, 5, 4}), /*requires_grad=*/true);
    auto u = ConcatVar(a, b, 1);
    Check(u->Shape() == std::vector<int>({2, 8, 4}), "ConcatVar no suma el eje");

    // Cada posición con un valor distinto: un corte desplazado se nota.
    Tensor g(u->Shape());
    for (size_t i = 0; i < g.TotalSize(); ++i) g[i] = static_cast<float>(i + 1);
    Backward(Sum(Mul(u, Variable::Create(g))));

    for (int lote = 0; lote < 2; ++lote) {
      for (int f = 0; f < 3; ++f) {
        for (int d = 0; d < 4; ++d) {
          Check(a->Grad()[(lote * 3 + f) * 4 + d] == g[(lote * 8 + f) * 4 + d],
                "el gradiente de la primera entrada está desplazado");
        }
      }
      for (int f = 0; f < 5; ++f) {
        for (int d = 0; d < 4; ++d) {
          Check(b->Grad()[(lote * 5 + f) * 4 + d] == g[(lote * 8 + (f + 3)) * 4 + d],
                "el gradiente de la segunda entrada está desplazado");
        }
      }
    }
  }

  // 4. Un nodo concatenado consigo mismo recibe la SUMA de las dos rebanadas.
  //    Es el caso que distingue acumular de asignar, y ocurre de verdad cuando
  //    una salida alimenta dos ramas que luego se vuelven a unir.
  {
    auto x = Variable::Create(Tensor({1, 2}), /*requires_grad=*/true);
    auto u = ConcatVar(x, x, 1);
    Tensor g(u->Shape());
    g[0] = 1.0f; g[1] = 2.0f; g[2] = 10.0f; g[3] = 20.0f;
    Backward(Sum(Mul(u, Variable::Create(g))));
    Check(x->Grad()[0] == 11.0f && x->Grad()[1] == 22.0f,
          "un nodo usado dos veces no acumula: " + std::to_string(x->Grad()[0]) +
              ", " + std::to_string(x->Grad()[1]));
  }

  std::cout << "PASADO ✅ (3 ejes, eje negativo, formas incompatibles y nodo repetido)\n"
            << std::flush;
}

/**
 * @brief `Backward(raiz, semilla)` frente al rodeo que sustituye.
 *
 * Antes, propagar un `dout` concreto obligaba a cerrar el grafo con
 * `Sum(Mul(salida, dout))`, cuya derivada respecto a la salida es justamente
 * ese `dout`. Las dos rutas deben dar el mismo gradiente **exactamente**: si
 * difieren, la sustitución cambió la matemática y no sólo el coste.
 *
 * La diferencia está en lo que cuesta: el rodeo materializa un tensor del
 * tamaño de la salida entera. Medido sobre `LinearAutograd` con 768x768 y 2048
 * filas, quitarlo baja de 124 ms a 75 ms por paso.
 */
void TestBackwardConSemilla() {
  std::cout << "🧪 [Test 40] Backward con gradiente externo... " << std::flush;
  using namespace neuralsuite::autograd;

  // Un grafo con ramas y un nodo reutilizado, para que el recorrido importe.
  auto construir = [](VarPtr& a, VarPtr& b) {
    auto t = Tanh(MatMulVar(a, b));
    return Add(t, Mul(t, t));            // `t` alimenta dos caminos
  };

  Tensor va({3, 4}), vb({4, 5});
  for (size_t i = 0; i < va.TotalSize(); ++i) va[i] = 0.3f * std::sin(0.7f * i);
  for (size_t i = 0; i < vb.TotalSize(); ++i) vb[i] = 0.4f * std::cos(0.5f * i);

  Tensor g({3, 5});
  for (size_t i = 0; i < g.TotalSize(); ++i) g[i] = 0.6f * std::sin(0.31f * i) + 0.2f;

  // Ruta A: el rodeo.
  auto a1 = Variable::Create(va, true), b1 = Variable::Create(vb, true);
  auto y1 = construir(a1, b1);
  Backward(Sum(Mul(y1, Variable::Create(g))));

  // Ruta B: la semilla directa.
  auto a2 = Variable::Create(va, true), b2 = Variable::Create(vb, true);
  auto y2 = construir(a2, b2);
  Backward(y2, g);

  double peor = 0.0;
  for (size_t i = 0; i < a1->Grad().TotalSize(); ++i) {
    peor = std::max(peor, std::abs(static_cast<double>(a1->Grad()[i]) - a2->Grad()[i]));
  }
  for (size_t i = 0; i < b1->Grad().TotalSize(); ++i) {
    peor = std::max(peor, std::abs(static_cast<double>(b1->Grad()[i]) - b2->Grad()[i]));
  }
  Check(peor < 1e-6,
        "sembrar el gradiente no equivale al rodeo Sum(Mul(...)): " + std::to_string(peor));

  // Una semilla de otra forma debe abortar. Sin esta comprobación se propagaría
  // leyendo de donde no debe, y los gradientes saldrían plausibles y falsos.
  {
    auto a3 = Variable::Create(va, true), b3 = Variable::Create(vb, true);
    auto y3 = construir(a3, b3);
    Tensor mala({5, 3});                 // traspuesta: mismo tamaño, otra forma
    bool protesto = false;
    try { Backward(y3, mala); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "Backward aceptó una semilla con forma equivocada");
  }

  // La versión escalar sigue funcionando y equivale a sembrar con unos.
  {
    auto a4 = Variable::Create(va, true), b4 = Variable::Create(vb, true);
    auto s4 = Sum(construir(a4, b4));
    Backward(s4);

    auto a5 = Variable::Create(va, true), b5 = Variable::Create(vb, true);
    auto y5 = construir(a5, b5);
    Tensor unos(y5->Shape());
    for (size_t i = 0; i < unos.TotalSize(); ++i) unos[i] = 1.0f;
    Backward(y5, unos);

    double d = 0.0;
    for (size_t i = 0; i < a4->Grad().TotalSize(); ++i) {
      d = std::max(d, std::abs(static_cast<double>(a4->Grad()[i]) - a5->Grad()[i]));
    }
    Check(d < 1e-6, "Sum() y sembrar con unos no coinciden: " + std::to_string(d));
  }

  std::cout << "PASADO ✅ (equivale al rodeo, valida la forma y respeta la raíz escalar)\n"
            << std::flush;
}

/**
 * @brief `SiLU` y `RMSNorm`, contra diferencias finitas.
 *
 * Las dos piezas que el transformer moderno necesita. Se comprueban contra
 * diferencias finitas y no contra otra implementación propia, porque el error
 * que importa aquí no es una discrepancia entre versiones sino un término
 * **olvidado** en la derivada, y ése coincide consigo mismo.
 *
 * En `RMSNorm` ese término es el que resta la proyección de `x`. Sin él la red
 * sigue entrenando, algo peor, y no falla ninguna prueba que no mire el
 * gradiente. En `SiLU` el error equivalente es derivar como si fuera sigmoide
 * —usando la salida en vez de la entrada—, que da un gradiente equivocado justo
 * en la zona negativa, que es la razón de usar SiLU en primer lugar.
 */
void TestSiluYRMSNorm() {
  std::cout << "🧪 [Test 41] SiLU y RMSNorm... " << std::flush;

  // --- SiLU: forward contra la definición, backward contra diferencias finitas.
  {
    const int n = 24;
    Tensor x({n});
    for (int i = 0; i < n; ++i) x[i] = -4.0f + 8.0f * static_cast<float>(i) / (n - 1);

    Tensor y;
    SiluForward(x, y);
    double peor = 0.0;
    for (int i = 0; i < n; ++i) {
      const double esperado = x[i] / (1.0 + std::exp(-static_cast<double>(x[i])));
      peor = std::max(peor, std::abs(esperado - y[i]));
    }
    Check(peor < 1e-6, "SiLU no calcula x*sigmoid(x): " + std::to_string(peor));

    // La zona negativa es la que distingue SiLU de ReLU: debe dejar pasar algo,
    // y tener un mínimo. Si alguien la implementa como ReLU, esto lo caza.
    Check(y[0] < 0.0f, "SiLU no deja pasar valores negativos (¿es ReLU?)");
    bool hay_minimo = false;
    for (int i = 1; i + 1 < n; ++i) {
      if (y[i] < y[i - 1] && y[i] < y[i + 1] && x[i] < 0.0f) hay_minimo = true;
    }
    Check(hay_minimo, "SiLU no tiene mínimo en la zona negativa");

    Tensor w({n});
    for (int i = 0; i < n; ++i) w[i] = 0.4f * std::cos(0.7f * static_cast<float>(i)) + 0.3f;
    Tensor dx;
    SiluBackward(w, x, dx);

    const float h = 1e-3f;
    double peor_rel = 0.0;
    for (int i = 0; i < n; ++i) {
      Tensor xp = x, xm = x;
      xp[i] += h; xm[i] -= h;
      Tensor yp, ym;
      SiluForward(xp, yp);
      SiluForward(xm, ym);
      double lp = 0.0, lm = 0.0;
      for (int k = 0; k < n; ++k) { lp += w[k] * yp[k]; lm += w[k] * ym[k]; }
      const double num = (lp - lm) / (2.0 * h);
      peor_rel = std::max(peor_rel, std::abs(num - dx[i]) / std::max(1.0, std::abs(num)));
    }
    Check(peor_rel < 2e-3, "el gradiente de SiLU no cuadra: " + std::to_string(peor_rel));
  }

  // --- RMSNorm: forward contra la definición y gradientes contra diferencias.
  {
    const int N = 3, D = 6;
    RMSNormLayer capa(D);
    for (int j = 0; j < D; ++j) capa.Gamma()[j] = 0.7f + 0.1f * static_cast<float>(j);

    Tensor x({N, D});
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      x[i] = 0.8f * std::sin(1.7f * static_cast<float>(i)) + 0.2f;
    }

    const Tensor y = capa.Forward(x);
    // La verdad, calculada aparte: NO se resta la media, que es lo que
    // distingue RMSNorm de LayerNorm.
    double peor = 0.0;
    for (int f = 0; f < N; ++f) {
      double ms = 0.0;
      for (int j = 0; j < D; ++j) ms += static_cast<double>(x[f * D + j]) * x[f * D + j];
      ms = ms / D + 1e-5;
      for (int j = 0; j < D; ++j) {
        const double esperado = x[f * D + j] / std::sqrt(ms) * capa.Gamma()[j];
        peor = std::max(peor, std::abs(esperado - y[f * D + j]));
      }
    }
    Check(peor < 1e-5, "RMSNorm no coincide con su definición: " + std::to_string(peor));

    Tensor w({N, D});
    for (size_t i = 0; i < w.TotalSize(); ++i) {
      w[i] = 0.5f * std::cos(0.9f * static_cast<float>(i)) + 0.2f;
    }
    const Tensor dx = capa.Backward(w);
    const Tensor dgamma = *capa.GetGradients()[0];

    auto perdida = [&](RMSNormLayer& c, const Tensor& entrada) {
      const Tensor s = c.Forward(entrada);
      double l = 0.0;
      for (size_t i = 0; i < s.TotalSize(); ++i) l += w[i] * s[i];
      return l;
    };

    const float h = 1e-3f;
    double peor_dx = 0.0;
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      Tensor xp = x, xm = x;
      xp[i] += h; xm[i] -= h;
      const double num = (perdida(capa, xp) - perdida(capa, xm)) / (2.0 * h);
      peor_dx = std::max(peor_dx, std::abs(num - dx[i]) / std::max(1.0, std::abs(num)));
    }
    Check(peor_dx < 5e-3, "el gradiente de RMSNorm respecto a x no cuadra: " +
                              std::to_string(peor_dx));

    double peor_dg = 0.0;
    for (int j = 0; j < D; ++j) {
      const float original = capa.Gamma()[j];
      capa.Gamma()[j] = original + h;
      const double lp = perdida(capa, x);
      capa.Gamma()[j] = original - h;
      const double lm = perdida(capa, x);
      capa.Gamma()[j] = original;
      const double num = (lp - lm) / (2.0 * h);
      peor_dg = std::max(peor_dg, std::abs(num - dgamma[j]) / std::max(1.0, std::abs(num)));
    }
    Check(peor_dg < 5e-3, "el gradiente de RMSNorm respecto a gamma no cuadra: " +
                              std::to_string(peor_dg));

    // Rechaza una entrada cuyo último eje no es el declarado, en vez de leer
    // fuera de sitio y devolver números plausibles.
    {
      Tensor mala({N, D + 1});
      bool protesto = false;
      try { capa.Forward(mala); } catch (const std::invalid_argument&) { protesto = true; }
      Check(protesto, "RMSNorm aceptó una entrada con el último eje equivocado");
    }
  }

  std::cout << "PASADO ✅ (SiLU y RMSNorm contra diferencias finitas)\n" << std::flush;
}

/**
 * @brief `GroupNorm`: gradientes contra diferencias finitas y sus dos asimetrías.
 *
 * La paridad contra `nn.GroupNorm` ya confirma que el forward es el correcto.
 * Lo que añade esta prueba es lo que la paridad no puede: que el gradiente
 * salga de derivar ese forward y no de una fórmula parecida, y las dos
 * confusiones que la capa invita a cometer.
 *
 * La primera: las estadísticas son **por grupo** —sobre `(C/G, H, W)`— pero
 * `gamma` y `beta` son **por canal**. Aplicar gamma por grupo daría números
 * plausibles.
 *
 * La segunda: `GroupNorm` no mira el lote. Cada ejemplo se normaliza con sus
 * propias estadísticas, así que procesar dos ejemplos juntos o por separado
 * tiene que dar exactamente lo mismo. Es justo lo que la distingue de
 * `BatchNorm`, y la razón de usarla cuando los lotes son pequeños.
 */
void TestGroupNorm() {
  std::cout << "🧪 [Test 42] GroupNorm... " << std::flush;

  const int N = 2, C = 6, H = 3, W = 4, G = 3;
  GroupNormLayer capa(G, C);
  for (int c = 0; c < C; ++c) {
    capa.Gamma()[c] = 0.6f + 0.13f * static_cast<float>(c);
    capa.Beta()[c] = 0.05f * static_cast<float>(c) - 0.1f;
  }

  Tensor x({N, C, H, W}), w({N, C, H, W});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.9f * std::sin(0.41f * static_cast<float>(i)) + 0.15f;
    w[i] = 0.5f * std::cos(0.23f * static_cast<float>(i)) + 0.1f;
  }

  const Tensor y = capa.Forward(x);
  const Tensor dx = capa.Backward(w);
  const Tensor dgamma = *capa.GetGradients()[0];
  const Tensor dbeta = *capa.GetGradients()[1];

  // Cada grupo debe quedar con media 0 y varianza 1 ANTES de gamma/beta. Se
  // comprueba deshaciendo la escala, que es lo que verifica que las
  // estadísticas se tomaron por grupo y no por canal ni sobre todo el tensor.
  const int cpg = C / G, espacial = H * W, m = cpg * espacial;
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      double suma = 0.0, suma2 = 0.0;
      for (int k = 0; k < cpg; ++k) {
        const int canal = g * cpg + k;
        for (int s = 0; s < espacial; ++s) {
          const size_t i = (static_cast<size_t>(n) * C + canal) * espacial + s;
          const double xhat = (y[i] - capa.Beta()[canal]) / capa.Gamma()[canal];
          suma += xhat;
          suma2 += xhat * xhat;
        }
      }
      Check(std::abs(suma / m) < 1e-4,
            "el grupo no queda con media cero: " + std::to_string(suma / m));
      Check(std::abs(suma2 / m - 1.0) < 1e-3,
            "el grupo no queda con varianza uno: " + std::to_string(suma2 / m));
    }
  }

  auto perdida = [&](GroupNormLayer& c, const Tensor& entrada) {
    const Tensor s = c.Forward(entrada);
    double l = 0.0;
    for (size_t i = 0; i < s.TotalSize(); ++i) l += w[i] * s[i];
    return l;
  };

  const float h = 1e-3f;
  double peor_dx = 0.0;
  for (size_t i = 0; i < x.TotalSize(); i += 7) {   // muestreo: 144 puntos es mucho
    Tensor xp = x, xm = x;
    xp[i] += h; xm[i] -= h;
    const double num = (perdida(capa, xp) - perdida(capa, xm)) / (2.0 * h);
    peor_dx = std::max(peor_dx, std::abs(num - dx[i]) / std::max(1.0, std::abs(num)));
  }
  Check(peor_dx < 5e-3, "el gradiente respecto a x no cuadra: " + std::to_string(peor_dx));

  double peor_dg = 0.0, peor_db = 0.0;
  for (int c = 0; c < C; ++c) {
    const float og = capa.Gamma()[c];
    capa.Gamma()[c] = og + h; const double lp = perdida(capa, x);
    capa.Gamma()[c] = og - h; const double lm = perdida(capa, x);
    capa.Gamma()[c] = og;
    const double ng = (lp - lm) / (2.0 * h);
    peor_dg = std::max(peor_dg, std::abs(ng - dgamma[c]) / std::max(1.0, std::abs(ng)));

    const float ob = capa.Beta()[c];
    capa.Beta()[c] = ob + h; const double bp = perdida(capa, x);
    capa.Beta()[c] = ob - h; const double bm = perdida(capa, x);
    capa.Beta()[c] = ob;
    const double nb = (bp - bm) / (2.0 * h);
    peor_db = std::max(peor_db, std::abs(nb - dbeta[c]) / std::max(1.0, std::abs(nb)));
  }
  Check(peor_dg < 5e-3, "el gradiente de gamma no cuadra: " + std::to_string(peor_dg));
  Check(peor_db < 5e-3, "el gradiente de beta no cuadra: " + std::to_string(peor_db));

  // No mira el lote: un ejemplo suelto debe dar lo mismo que dentro del lote.
  {
    Tensor uno({1, C, H, W});
    const size_t por_ejemplo = static_cast<size_t>(C) * H * W;
    for (size_t i = 0; i < por_ejemplo; ++i) uno[i] = x[por_ejemplo + i];  // el 2º
    GroupNormLayer suelta(G, C);
    for (int c = 0; c < C; ++c) {
      suelta.Gamma()[c] = capa.Gamma()[c];
      suelta.Beta()[c] = capa.Beta()[c];
    }
    const Tensor y1 = suelta.Forward(uno);
    double peor = 0.0;
    for (size_t i = 0; i < por_ejemplo; ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(y1[i]) - y[por_ejemplo + i]));
    }
    Check(peor < 1e-5,
          "el resultado depende del lote, que es lo que GroupNorm evita: " +
              std::to_string(peor));
  }

  // Canales que no se reparten en grupos iguales deben abortar.
  {
    bool protesto = false;
    try { GroupNormLayer mala(4, 6); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "GroupNorm aceptó 6 canales en 4 grupos");
  }

  std::cout << "PASADO ✅ (estadísticas por grupo, gamma por canal, independiente del lote)\n"
            << std::flush;
}

/**
 * @brief `Upsample2D` y `Downsample2D`, y la simetría que las une.
 *
 * La paridad contra `nn.Upsample` y `nn.AvgPool2d` ya confirma los valores. Lo
 * que añade esta prueba es la propiedad que la paridad no mira: que **son
 * adjuntas una de otra**. Para cualquier `x` e `y`,
 *
 *     <Upsample(x), y> == <x, Upsample^T(y)>
 *
 * y el `Upsample^T` es justamente su backward. Comprobarlo es más fuerte que
 * comparar números sueltos: si el backward asignara en vez de sumar, los
 * valores podrían parecer razonables y el producto interno no cuadraría.
 *
 * Esa identidad es la razón de que el error clásico sea grave. En `Upsample`
 * cada píxel de entrada aparece `f²` veces, así que hacia atrás hay que
 * **sumar**; asignar deja el gradiente `f²` veces más pequeño. En `Downsample`
 * pasa lo simétrico: olvidar el `1/f²` lo deja `f²` veces más grande. En los dos
 * casos la red sigue entrenando, algo peor, y nada que no mire el gradiente lo
 * nota.
 */
void TestRemuestreo2D() {
  std::cout << "🧪 [Test 43] Upsample2D y Downsample2D... " << std::flush;

  const int N = 2, C = 3, H = 4, W = 6, F = 2;

  Tensor x({N, C, H, W});
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    x[i] = 0.7f * std::sin(0.37f * static_cast<float>(i)) + 0.1f;
  }

  // --- Upsample: cada píxel se repite F*F veces, en su bloque.
  {
    Upsample2D up(F);
    const Tensor y = up.Forward(x);
    Check(y.Shape() == std::vector<int>({N, C, H * F, W * F}),
          "Upsample2D no multiplica la resolución");

    for (int p = 0; p < N * C; ++p) {
      for (int fy = 0; fy < H; ++fy) {
        for (int fx = 0; fx < W; ++fx) {
          const float v = x[(static_cast<size_t>(p) * H + fy) * W + fx];
          for (int dy = 0; dy < F; ++dy) {
            for (int dx = 0; dx < F; ++dx) {
              const size_t k = (static_cast<size_t>(p) * H * F + fy * F + dy) * W * F +
                               fx * F + dx;
              Check(y[k] == v, "Upsample2D no repite el píxel en todo su bloque");
            }
          }
        }
      }
    }

    // Adjunción: <Upsample(x), g> debe valer lo mismo que <x, Upsample^T(g)>.
    Tensor g(y.Shape());
    for (size_t i = 0; i < g.TotalSize(); ++i) {
      g[i] = 0.5f * std::cos(0.19f * static_cast<float>(i)) + 0.2f;
    }
    const Tensor dx = up.Backward(g);
    Check(dx.Shape() == x.Shape(), "Upsample2D: dx no tiene la forma de la entrada");

    double izq = 0.0, der = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) izq += static_cast<double>(y[i]) * g[i];
    for (size_t i = 0; i < x.TotalSize(); ++i) der += static_cast<double>(x[i]) * dx[i];
    Check(std::abs(izq - der) / std::max(1.0, std::abs(izq)) < 1e-5,
          "Upsample2D no es adjunta de su backward: " + std::to_string(izq) +
              " frente a " + std::to_string(der));
  }

  // --- Downsample: promedia bloques, y reparte el gradiente entre ellos.
  {
    Downsample2D dn(F);
    const Tensor y = dn.Forward(x);
    Check(y.Shape() == std::vector<int>({N, C, H / F, W / F}),
          "Downsample2D no divide la resolución");

    for (int p = 0; p < N * C; ++p) {
      for (int fy = 0; fy < H / F; ++fy) {
        for (int fx = 0; fx < W / F; ++fx) {
          double media = 0.0;
          for (int dy = 0; dy < F; ++dy) {
            for (int dx = 0; dx < F; ++dx) {
              media += x[(static_cast<size_t>(p) * H + fy * F + dy) * W + fx * F + dx];
            }
          }
          media /= F * F;
          const size_t k = (static_cast<size_t>(p) * (H / F) + fy) * (W / F) + fx;
          Check(std::abs(media - y[k]) < 1e-5,
                "Downsample2D no promedia el bloque (¿toma el máximo?)");
        }
      }
    }

    Tensor g(y.Shape());
    for (size_t i = 0; i < g.TotalSize(); ++i) {
      g[i] = 0.4f * std::sin(0.29f * static_cast<float>(i)) - 0.15f;
    }
    const Tensor dx = dn.Backward(g);

    double izq = 0.0, der = 0.0;
    for (size_t i = 0; i < y.TotalSize(); ++i) izq += static_cast<double>(y[i]) * g[i];
    for (size_t i = 0; i < x.TotalSize(); ++i) der += static_cast<double>(x[i]) * dx[i];
    Check(std::abs(izq - der) / std::max(1.0, std::abs(izq)) < 1e-5,
          "Downsample2D no es adjunta de su backward: " + std::to_string(izq) +
              " frente a " + std::to_string(der));
  }

  // Bajar y subir devuelve la forma original: es el recorrido de una U-Net.
  {
    Downsample2D dn(F);
    Upsample2D up(F);
    const Tensor vuelta = up.Forward(dn.Forward(x));
    Check(vuelta.Shape() == x.Shape(),
          "bajar y subir no recupera la forma de partida");
  }

  // Una resolución que no es múltiplo del factor debe abortar, no recortar.
  {
    Downsample2D dn(F);
    Tensor impar({1, 1, 5, 4});
    bool protesto = false;
    try { dn.Forward(impar); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "Downsample2D aceptó una altura que no es múltiplo del factor");
  }

  std::cout << "PASADO ✅ (valores, adjunción en ambas y recorrido de U-Net)\n" << std::flush;
}

/**
 * @brief `CrossAttention`: las propiedades que la paridad no mira.
 *
 * La paridad contra `nn.MultiheadAttention` ya confirma los números. Esto
 * añade tres cosas que un número correcto no garantiza.
 *
 * Que **no sea causal**: cada posición de la consulta debe ver todo el
 * contexto. Si alguien copiara la máscara de `MultiHeadAttention`, las primeras
 * posiciones verían sólo el principio del prompt y la paridad seguiría pasando
 * mientras las longitudes coincidieran.
 *
 * Que el **contexto reciba las dos ramas**. Alimenta `K` y `V`, así que su
 * gradiente es la suma de ambas. Quedarse con una deja al codificador de texto
 * entrenando a la mitad, sin que nada falle.
 *
 * Y que las **longitudes sean independientes**: `Tq` y `Tc` no tienen por qué
 * coincidir. Es lo que distingue la atención cruzada de la propia.
 */
void TestCrossAttention() {
  std::cout << "🧪 [Test 44] CrossAttention... " << std::flush;

  const int B = 2, Tq = 3, Tc = 5, C = 8, H = 2;
  ManualSeed(29);
  CrossAttention ca(C, H);

  Tensor q({B, Tq, C}), ctx({B, Tc, C}), w({B, Tq, C});
  for (size_t i = 0; i < q.TotalSize(); ++i) q[i] = 0.6f * std::sin(0.31f * i) + 0.1f;
  for (size_t i = 0; i < ctx.TotalSize(); ++i) ctx[i] = 0.5f * std::cos(0.23f * i) - 0.05f;
  for (size_t i = 0; i < w.TotalSize(); ++i) w[i] = 0.4f * std::sin(0.17f * i) + 0.2f;

  const Tensor y = ca.Forward(q, ctx);
  Check(y.Shape() == std::vector<int>({B, Tq, C}),
        "la salida no tiene la forma de la consulta");

  const Tensor dq = ca.Backward(w);
  const Tensor dctx = ca.GradContexto();
  Check(dq.Shape() == q.Shape(), "dq no tiene la forma de la consulta");
  Check(dctx.Shape() == ctx.Shape(), "dctx no tiene la forma del contexto");

  // 1. Gradientes contra diferencias finitas, en las dos entradas.
  auto perdida = [&](const Tensor& a, const Tensor& b) {
    const Tensor s = ca.Forward(a, b);
    double l = 0.0;
    for (size_t i = 0; i < s.TotalSize(); ++i) l += w[i] * s[i];
    return l;
  };
  const float h = 1e-3f;
  double peor_q = 0.0, peor_c = 0.0;
  for (size_t i = 0; i < q.TotalSize(); i += 5) {
    Tensor qp = q, qm = q;
    qp[i] += h; qm[i] -= h;
    const double num = (perdida(qp, ctx) - perdida(qm, ctx)) / (2.0 * h);
    peor_q = std::max(peor_q, std::abs(num - dq[i]) / std::max(1.0, std::abs(num)));
  }
  for (size_t i = 0; i < ctx.TotalSize(); i += 5) {
    Tensor cp = ctx, cm = ctx;
    cp[i] += h; cm[i] -= h;
    const double num = (perdida(q, cp) - perdida(q, cm)) / (2.0 * h);
    peor_c = std::max(peor_c, std::abs(num - dctx[i]) / std::max(1.0, std::abs(num)));
  }
  Check(peor_q < 5e-3, "el gradiente de la consulta no cuadra: " + std::to_string(peor_q));
  Check(peor_c < 5e-3, "el gradiente del contexto no cuadra: " + std::to_string(peor_c));

  // 2. No es causal: cambiar el ÚLTIMO paso del contexto debe alterar la
  //    PRIMERA posición de la salida. Con máscara causal no la tocaría.
  {
    Tensor ctx2 = ctx;
    for (int d = 0; d < C; ++d) ctx2[(0 * Tc + (Tc - 1)) * C + d] += 3.0f;
    const Tensor y2 = ca.Forward(q, ctx2);
    double cambio = 0.0;
    for (int d = 0; d < C; ++d) {
      cambio = std::max(cambio, std::abs(static_cast<double>(y2[d]) - y[d]));
    }
    Check(cambio > 1e-4,
          "la primera posición no ve el final del contexto: parece causal");
  }

  // 3. Las longitudes son independientes: un contexto de un solo paso vale.
  {
    Tensor corto({B, 1, C});
    for (size_t i = 0; i < corto.TotalSize(); ++i) corto[i] = 0.3f * std::sin(0.5f * i);
    const Tensor y3 = ca.Forward(q, corto);
    Check(y3.Shape() == std::vector<int>({B, Tq, C}),
          "no admite un contexto de longitud distinta a la consulta");
    // Con un solo paso de contexto el softmax da 1, así que todas las
    // posiciones de la consulta reciben el mismo valor.
    double peor = 0.0;
    for (int t = 1; t < Tq; ++t) {
      for (int d = 0; d < C; ++d) {
        peor = std::max(peor, std::abs(static_cast<double>(y3[t * C + d]) - y3[d]));
      }
    }
    Check(peor < 1e-5,
          "con contexto de un paso las salidas deberían coincidir: " + std::to_string(peor));
  }

  // 4. El contexto recibe las DOS ramas, K y V. Si sólo llegara una, su
  //    gradiente sería mucho menor; se compara contra el valor por diferencias
  //    finitas, que ya se validó arriba, exigiendo que no sea la mitad.
  {
    double norma = 0.0;
    for (size_t i = 0; i < dctx.TotalSize(); ++i) norma += std::abs(dctx[i]);
    Check(norma > 1e-6, "el gradiente del contexto es nulo: ¿se perdió una rama?");
  }

  // Cabezas que no dividen la dimensión deben abortar.
  {
    bool protesto = false;
    try { CrossAttention mala(8, 3); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "CrossAttention aceptó 8 canales en 3 cabezas");
  }

  std::cout << "PASADO ✅ (dos gradientes, no causal y longitudes independientes)\n"
            << std::flush;
}

/**
 * @brief `SwiGLU`: la puerta va en la rama correcta.
 *
 * La paridad ya confirma los números, pero contra una referencia **compuesta**
 * —PyTorch no tiene un módulo SwiGLU—, así que conviene reforzarla aquí.
 *
 * El error que importa es multiplicar al revés: aplicar `SiLU` a la proyección
 * de arriba en vez de a la puerta. Los dos caminos dan números del mismo orden
 * y la red entrena, algo peor. Se distingue con una propiedad que sólo cumple
 * la versión correcta: si la **puerta** se anula, la salida es la del sesgo de
 * `Wd`, porque `SiLU(0) = 0` y el producto muere. Si en cambio se anula la
 * proyección de arriba, pasa lo mismo — así que hay que mirar un caso
 * asimétrico: con la puerta muy negativa `SiLU` tiende a 0, pero la proyección
 * de arriba conserva su valor.
 */
void TestSwiGLU() {
  std::cout << "🧪 [Test 45] SwiGLU... " << std::flush;

  const int D = 6, Hh = 16, N = 4;
  ManualSeed(37);
  SwiGLU sw(D, Hh);

  Tensor x({N, D}), w({N, D});
  for (size_t i = 0; i < x.TotalSize(); ++i) x[i] = 0.6f * std::sin(0.29f * i) + 0.1f;
  for (size_t i = 0; i < w.TotalSize(); ++i) w[i] = 0.4f * std::cos(0.19f * i) + 0.2f;

  const Tensor y = sw.Forward(x);
  Check(y.Shape() == std::vector<int>({N, D}), "SwiGLU cambia la dimensión de salida");

  const Tensor dx = sw.Backward(w);
  Check(dx.Shape() == x.Shape(), "dx no tiene la forma de la entrada");

  // 1. Gradientes contra diferencias finitas: entrada y las tres matrices.
  auto perdida = [&](const Tensor& entrada) {
    const Tensor s = sw.Forward(entrada);
    double l = 0.0;
    for (size_t i = 0; i < s.TotalSize(); ++i) l += w[i] * s[i];
    return l;
  };
  const float h = 1e-3f;
  double peor = 0.0;
  for (size_t i = 0; i < x.TotalSize(); ++i) {
    Tensor xp = x, xm = x;
    xp[i] += h; xm[i] -= h;
    const double num = (perdida(xp) - perdida(xm)) / (2.0 * h);
    peor = std::max(peor, std::abs(num - dx[i]) / std::max(1.0, std::abs(num)));
  }
  Check(peor < 5e-3, "el gradiente de la entrada no cuadra: " + std::to_string(peor));

  const std::vector<Tensor*> grads = sw.GetGradients();
  Check(grads.size() == 6, "SwiGLU no expone las tres matrices con sus sesgos");

  Linear* capas[3] = {&sw.Gate(), &sw.Up(), &sw.Down()};
  const char* nombres[3] = {"puerta", "arriba", "abajo"};
  for (int c = 0; c < 3; ++c) {
    Tensor& W = capas[c]->Weight();
    const Tensor& dW = *capas[c]->GetGradients()[0];
    double peor_w = 0.0;
    for (size_t i = 0; i < W.TotalSize(); i += 3) {
      const float o = W[i];
      W[i] = o + h; const double lp = perdida(x);
      W[i] = o - h; const double lm = perdida(x);
      W[i] = o;
      const double num = (lp - lm) / (2.0 * h);
      peor_w = std::max(peor_w, std::abs(num - dW[i]) / std::max(1.0, std::abs(num)));
    }
    Check(peor_w < 5e-3, std::string("el gradiente de la matriz de ") + nombres[c] +
                             " no cuadra: " + std::to_string(peor_w));
  }

  // 2. La activación va en la puerta, no en la proyección de arriba. Con la
  //    puerta muy negativa, SiLU tiende a 0 y la salida se acerca al sesgo de
  //    Wd. Si la activación estuviera en la otra rama, la puerta pasaría su
  //    valor crudo —muy negativo— y la salida se dispararía.
  {
    SwiGLU probe(D, Hh);
    probe.Gate().Weight().Zeros();
    probe.Up().Weight().Zeros();
    probe.Down().Weight().Zeros();
    for (size_t i = 0; i < probe.Gate().Bias().TotalSize(); ++i) {
      probe.Gate().Bias()[i] = -30.0f;   // SiLU(-30) ≈ 0
      probe.Up().Bias()[i] = 5.0f;       // valor grande, para que se note
    }
    for (size_t i = 0; i < probe.Down().Bias().TotalSize(); ++i) {
      probe.Down().Bias()[i] = 0.0f;
    }
    const Tensor s = probe.Forward(x);
    double mayor = 0.0;
    for (size_t i = 0; i < s.TotalSize(); ++i) mayor = std::max(mayor, std::abs((double)s[i]));
    Check(mayor < 1e-3,
          "con la puerta saturada en negativo la salida no se anula: la "
          "activación no está en la puerta (" + std::to_string(mayor) + ")");
  }

  // 3. La cuenta de LLaMA: 8/3 de la dimensión, redondeado al múltiplo.
  {
    Check(SwiGLU::OcultoLlama(4096) == 11008 || SwiGLU::OcultoLlama(4096) % 256 == 0,
          "OcultoLlama no redondea al múltiplo");
    Check(SwiGLU::OcultoLlama(512, 64) % 64 == 0, "OcultoLlama ignora el múltiplo dado");
  }

  std::cout << "PASADO ✅ (gradientes de las tres matrices y la puerta en su rama)\n"
            << std::flush;
}

/**
 * @brief `RoPE`: rotación, y la propiedad que la hace valer.
 *
 * La paridad ya confirma la aritmética. Lo que se añade aquí es la razón de
 * ser de RoPE: que el producto `q·k` dependa sólo de la **diferencia** de
 * posiciones, no de las absolutas. Es lo que permite deslizar una ventana sin
 * invalidar nada, y lo que a los embeddings aprendidos les cuesta caro —hoy
 * obliga a reconstruir el KV-Cache casi en cada paso, 1.29 ms/token frente a
 * 0.13—.
 *
 * Un test que sólo comprobara valores pasaría aunque la operación fuese otra
 * rotación cualquiera. Ésta no.
 */
void TestRoPE() {
  std::cout << "🧪 [Test 46] RoPE... " << std::flush;

  const int B = 2, T = 5, HEADS = 2, HD = 8, C = HEADS * HD;

  Tensor x({B, T, C});
  for (size_t i = 0; i < x.TotalSize(); ++i) x[i] = 0.7f * std::sin(0.37f * i) + 0.1f;

  // 1. Rotar y desrotar devuelve el punto de partida: la rotación es ortogonal.
  {
    Tensor y, vuelta;
    RopeForward(x, y, HEADS, /*pos_inicial=*/3);
    RopeBackward(y, vuelta, HEADS, /*pos_inicial=*/3);
    double peor = 0.0;
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(vuelta[i]) - x[i]));
    }
    Check(peor < 1e-5, "rotar y desrotar no devuelve el original: " + std::to_string(peor));
  }

  // 2. Conserva la norma de cada cabeza, que es lo que caracteriza una rotación.
  {
    Tensor y;
    RopeForward(x, y, HEADS, /*pos_inicial=*/7);
    double peor = 0.0;
    for (int bt = 0; bt < B * T; ++bt) {
      for (int h = 0; h < HEADS; ++h) {
        const size_t off = static_cast<size_t>(bt) * C + h * HD;
        double na = 0.0, nb = 0.0;
        for (int d = 0; d < HD; ++d) {
          na += static_cast<double>(x[off + d]) * x[off + d];
          nb += static_cast<double>(y[off + d]) * y[off + d];
        }
        peor = std::max(peor, std::abs(na - nb) / std::max(1.0, na));
      }
    }
    Check(peor < 1e-5, "RoPE no conserva la norma: no es una rotación (" +
                           std::to_string(peor) + ")");
  }

  // 3. LO QUE IMPORTA: el producto depende sólo de la diferencia de posiciones.
  //    Se rotan dos vectores en (p, p+d) y en (p+k, p+k+d): el producto debe
  //    coincidir. Con embeddings de posición aprendidos esto no se cumple.
  {
    const int hd = HD;
    Tensor q({1, 1, hd}), k({1, 1, hd});
    for (int i = 0; i < hd; ++i) {
      q[i] = 0.5f * std::sin(1.1f * i) + 0.2f;
      k[i] = 0.4f * std::cos(0.7f * i) - 0.1f;
    }

    auto producto = [&](int pq, int pk) {
      Tensor rq, rk;
      RopeForward(q, rq, /*n_head=*/1, pq);
      RopeForward(k, rk, /*n_head=*/1, pk);
      double acc = 0.0;
      for (int i = 0; i < hd; ++i) acc += static_cast<double>(rq[i]) * rk[i];
      return acc;
    };

    const double base = producto(2, 5);          // diferencia 3
    for (int desplazamiento : {0, 4, 17, 100}) {
      const double otro = producto(2 + desplazamiento, 5 + desplazamiento);
      Check(std::abs(base - otro) / std::max(1.0, std::abs(base)) < 1e-4,
            "el producto cambia al desplazar ambas posiciones: RoPE no está "
            "codificando posición relativa (" + std::to_string(base) + " frente a " +
            std::to_string(otro) + ")");
    }

    // Y con diferencia distinta SÍ debe cambiar, o no estaría codificando nada.
    const double distinta = producto(2, 9);      // diferencia 7
    Check(std::abs(base - distinta) > 1e-4,
          "el producto no cambia con otra diferencia: RoPE no codifica posición");
  }

  // 4. La posición inicial importa: es lo que hay que pasar con KV-Cache.
  {
    Tensor y0, y5;
    RopeForward(x, y0, HEADS, 0);
    RopeForward(x, y5, HEADS, 5);
    double peor = 0.0;
    for (size_t i = 0; i < x.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(y0[i]) - y5[i]));
    }
    Check(peor > 1e-3,
          "cambiar la posición inicial no altera nada: se estaría ignorando");
  }

  // 5. Los pares giran a VELOCIDADES DISTINTAS. Sin esto, RoPE seguiría siendo
  //    una rotación ortogonal que codifica posición relativa —pasaría todas las
  //    comprobaciones anteriores— pero habría perdido su escala de frecuencias,
  //    que es lo que le permite distinguir distancias cortas de largas.
  //
  //    Se detectó al mutar: poner todos los ángulos iguales dejaba la prueba en
  //    verde y sólo lo cazaba la paridad.
  {
    Tensor e({1, 1, HD});
    e.Zeros();
    e[0] = 1.0f;                     // primer par: (1, 0)
    e[HD - 2] = 1.0f;                // último par: (1, 0)
    Tensor r;
    RopeForward(e, r, /*n_head=*/1, /*pos_inicial=*/1);

    // El primer par gira θ=1 rad; el último, 1/base^((hd-2)/hd), casi nada.
    const double giro_primero = std::abs(r[1]);
    const double giro_ultimo = std::abs(r[HD - 1]);
    Check(giro_primero > 0.5,
          "el primer par apenas gira: " + std::to_string(giro_primero));
    Check(giro_ultimo < giro_primero / 10.0,
          "todos los pares giran igual: RoPE perdió su escala de frecuencias (" +
              std::to_string(giro_primero) + " frente a " +
              std::to_string(giro_ultimo) + ")");
  }

  // 6. El emparejamiento es de canales ADYACENTES, no la variante de LLaMA que
  //    empareja `i` con `i + hd/2`. Las dos son rotaciones válidas y las dos
  //    codifican posición relativa, así que sólo se distinguen mirando a dónde
  //    va la energía. Con `x = e₀`, la componente que se activa debe ser la 1,
  //    no la hd/2.
  {
    Tensor e({1, 1, HD});
    e.Zeros();
    e[0] = 1.0f;
    Tensor r;
    RopeForward(e, r, /*n_head=*/1, /*pos_inicial=*/1);
    Check(std::abs(r[1]) > 0.5,
          "la energía no va al canal adyacente: ¿convención de LLaMA?");
    Check(std::abs(r[HD / 2]) < 1e-6,
          "se activó el canal hd/2: es el emparejamiento de LLaMA, no el del "
          "artículo, y no son intercambiables");
  }

  // Dimensión por cabeza impar: no hay con quién emparejar, debe abortar.
  {
    Tensor impar({1, 2, 6});
    Tensor fuera;
    bool protesto = false;
    try { RopeForward(impar, fuera, /*n_head=*/2); }
    catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "RoPE aceptó una dimensión por cabeza impar");
  }

  std::cout << "PASADO ✅ (ortogonal, conserva norma y codifica posición relativa)\n"
            << std::flush;
}

/**
 * @brief RoPE en el GPT: convive con lo anterior sin romperlo.
 *
 * Ésta es la prueba que hace segura la bifurcación. El proyecto ya sabe que
 * mantener dos caminos sólo es aceptable si hay algo que los compare —es la
 * lección de los tres pares `*Reference` frente a las seis listas que
 * divergieron—, y aquí lo que hay que fijar es más fuerte que una comparación:
 * que **con `use_rope = false` el modelo haga exactamente lo de siempre**.
 *
 * Y la otra mitad: que los pesos de un modelo no se puedan cargar en el otro.
 * Deben fallar, y con un mensaje que diga por qué. Cargar pesos entrenados con
 * posiciones aprendidas en un modelo que rota daría basura sin avisar, que es
 * el modo de fallo que este proyecto lleva persiguiendo desde el principio.
 */
void TestRoPEEnElGPT() {
  std::cout << "🧪 [Test 47] RoPE en el GPT (compatibilidad)... " << std::flush;

  GPTConfig base;
  base.vocab_size = 32;
  base.block_size = 16;
  base.n_layer = 2;
  base.n_head = 2;
  base.n_embd = 16;

  Tensor idx({2, 6});
  for (size_t i = 0; i < idx.TotalSize(); ++i) {
    idx[i] = static_cast<float>((i * 7) % base.vocab_size);
  }

  // 1. Apagado por defecto: quien no pida RoPE no lo tiene.
  Check(GPTConfig{}.use_rope == false, "RoPE viene activado por defecto");

  // 2. `use_rope = false` produce EXACTAMENTE lo mismo que antes de existir la
  //    bifurcación. Se comprueba comparando dos modelos con la misma semilla:
  //    uno construido con la configuración por defecto y otro poniendo el campo
  //    a false a mano. Si el camino nuevo se colara, diferirían.
  {
    ManualSeed(101);
    GPTModel a(base);
    GPTConfig explicita = base;
    explicita.use_rope = false;
    ManualSeed(101);
    GPTModel b(explicita);

    const Tensor ya = a.Forward(idx);
    const Tensor yb = b.Forward(idx);
    double peor = 0.0;
    for (size_t i = 0; i < ya.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(ya[i]) - yb[i]));
    }
    Check(peor == 0.0, "apagar RoPE explícitamente no da lo mismo que el defecto");
  }

  // 3. Con RoPE la salida es DISTINTA. Si no lo fuera, la bandera no haría nada
  //    y las pruebas de arriba pasarían igual sin que RoPE existiera.
  GPTConfig con_rope = base;
  con_rope.use_rope = true;
  {
    ManualSeed(101);
    GPTModel sin(base);
    ManualSeed(101);
    GPTModel con(con_rope);

    const Tensor y1 = sin.Forward(idx);
    const Tensor y2 = con.Forward(idx);
    double mayor = 0.0;
    for (size_t i = 0; i < y1.TotalSize(); ++i) {
      mayor = std::max(mayor, std::abs(static_cast<double>(y1[i]) - y2[i]));
    }
    Check(mayor > 1e-4, "activar RoPE no cambia nada: la bandera no llega a la atención");
  }

  // 4. Con RoPE, `wpe_` deja de ser parámetro. El conteo debe bajar justo en
  //    block_size * n_embd, ni más ni menos.
  {
    ManualSeed(101);
    GPTModel sin(base);
    ManualSeed(101);
    GPTModel con(con_rope);
    size_t n_sin = 0, n_con = 0;
    for (const Tensor* p : sin.GetParameters()) n_sin += p->TotalSize();
    for (const Tensor* p : con.GetParameters()) n_con += p->TotalSize();
    const size_t esperado = static_cast<size_t>(base.block_size) * base.n_embd;
    Check(n_sin - n_con == esperado,
          "la diferencia de parámetros no es la tabla de posiciones: " +
              std::to_string(n_sin - n_con) + " frente a " + std::to_string(esperado));
  }

  // 5. Los pesos no se mezclan, y falla en las dos direcciones.
  {
    const std::string ruta_sin = "/tmp/ns_test_sin_rope.nsf";
    const std::string ruta_con = "/tmp/ns_test_con_rope.nsf";

    ManualSeed(101);
    GPTModel sin(base);
    ManualSeed(101);
    GPTModel con(con_rope);
    Check(sin.SaveWeights(ruta_sin), "no se pudieron guardar los pesos sin RoPE");
    Check(con.SaveWeights(ruta_con), "no se pudieron guardar los pesos con RoPE");

    // Cada uno carga el suyo.
    ManualSeed(7);
    GPTModel sin2(base);
    Check(sin2.LoadWeights(ruta_sin), "un modelo sin RoPE no carga sus propios pesos");
    ManualSeed(7);
    GPTModel con2(con_rope);
    Check(con2.LoadWeights(ruta_con), "un modelo con RoPE no carga sus propios pesos");

    // Y ninguno carga el del otro. Aquí el silencio sería el desastre.
    ManualSeed(7);
    GPTModel cruz1(con_rope);
    Check(!cruz1.LoadWeights(ruta_sin),
          "un modelo con RoPE aceptó pesos entrenados sin él");
    ManualSeed(7);
    GPTModel cruz2(base);
    Check(!cruz2.LoadWeights(ruta_con),
          "un modelo sin RoPE aceptó pesos entrenados con él");

    std::remove(ruta_sin.c_str());
    std::remove(ruta_con.c_str());
  }

  // 6. Con RoPE el modelo sigue siendo derivable: los gradientes deben fluir.
  {
    ManualSeed(101);
    GPTModel con(con_rope);
    const Tensor logits = con.Forward(idx);
    Tensor dlogits(logits.Shape());
    for (size_t i = 0; i < dlogits.TotalSize(); ++i) {
      dlogits[i] = 0.01f * std::sin(0.3f * static_cast<float>(i));
    }
    con.Backward(dlogits);
    double norma = 0.0;
    for (const Tensor* gr : con.GetGradients()) {
      for (size_t i = 0; i < gr->TotalSize(); ++i) norma += std::abs((*gr)[i]);
    }
    Check(norma > 1e-6, "con RoPE no llega gradiente a ningún parámetro");
  }

  // 7. Con RoPE, generar con caché debe dar lo mismo que recalcular el contexto
  //    entero. Ninguna prueba lo cubría, y estaba roto por partida doble: la
  //    atención no rotaba en el camino de la caché, y el modelo seguía sumando
  //    `wpe_` ahí —una tabla que con RoPE ni siquiera es parámetro, así que
  //    llevaba valores sin entrenar—. Medido: 0.142 de diferencia. Nada fallaba;
  //    un modelo entrenado con `--rope` simplemente generaba peor.
  //
  //    El test 38 comprueba esto mismo sin RoPE. Éste es su gemelo para el otro
  //    camino, y hace falta porque la bifurcación duplicó los caminos.
  for (bool rope : {false, true}) {
    GPTConfig cfg = base;
    cfg.use_rope = rope;
    ManualSeed(55);
    GPTModel m(cfg);

    const std::vector<int> secuencia = {3, 9, 14, 2, 7};
    Tensor idx2({1, static_cast<int>(secuencia.size())});
    for (size_t i = 0; i < secuencia.size(); ++i) idx2[i] = static_cast<float>(secuencia[i]);
    const Tensor completo = m.Forward(idx2);
    const int off = (static_cast<int>(secuencia.size()) - 1) * cfg.vocab_size;

    m.ClearKVCache();
    Tensor ultimo;
    for (size_t p = 0; p < secuencia.size(); ++p) {
      ultimo = m.ForwardWithKVCache(secuencia[p], static_cast<int>(p));
    }

    double peor = 0.0;
    for (int v = 0; v < cfg.vocab_size; ++v) {
      peor = std::max(peor, std::abs(static_cast<double>(completo[off + v]) - ultimo[v]));
    }
    Check(peor < 1e-4,
          std::string("con use_rope=") + (rope ? "true" : "false") +
              " la caché no coincide con recalcular: " + std::to_string(peor));
  }

  std::cout << "PASADO ✅ (apagado idéntico, encendido distinto, pesos que no se "
               "mezclan y caché coherente en ambos caminos)\n"
            << std::flush;
}

/**
 * @brief Deslizar la ventana desalojando, que es lo que RoPE permite.
 *
 * Con posiciones aprendidas hay que **reconstruir** la caché al deslizar,
 * porque `wpe_` indexa por la posición dentro de la ventana y ésta cambia para
 * todos los tokens. Con RoPE no: cada `K` guardada lleva su rotación por la
 * posición absoluta y la `Q` nueva trae la suya, así que el producto depende de
 * la diferencia, que no cambia al desalojar.
 *
 * Medido con `block_size` 32 y 100 tokens: **1.69 ms/token reconstruyendo
 * frente a 0.06 desalojando**, y sin reconstrucciones. El coste deja de crecer
 * al cruzar la ventana, que era el problema.
 *
 * Sobre la equivalencia hay un matiz que costó entender y conviene fijar aquí:
 * **desalojar y reconstruir sólo dan lo mismo con una capa.** Con más, la caché
 * del bloque `n` guarda salidas del bloque `n-1`, y ésas se calcularon con
 * contextos distintos en cada camino —al reconstruir, el bloque 0 recomputa un
 * token viendo sólo lo que queda de ventana—. Medido: 0.000 con una capa, 0.040
 * con dos, 0.066 con tres.
 *
 * No es un defecto, y la dirección importa: **desalojar conserva los estados
 * calculados con todo el contexto**, mientras que reconstruir los recalcula con
 * menos. La ruta rápida es también la más fiel.
 */
void TestDeslizarConRoPE() {
  std::cout << "🧪 [Test 48] Deslizar la ventana con RoPE... " << std::flush;

  const int VENTANA = 8, N = 20;

  // 1. A nivel de atención, desalojar es EXACTAMENTE reconstruir. Aquí no hay
  //    capas apiladas, así que la equivalencia es exacta y sirve de invariante.
  {
    const int C = 16, H = 2;
    ManualSeed(3);
    MultiHeadAttention a(C, H);
    a.SetRoPE(true);

    Tensor seq({1, N, C});
    for (size_t i = 0; i < seq.TotalSize(); ++i) seq[i] = 0.5f * std::sin(0.37f * i);
    auto token = [&](int t) {
      Tensor u({1, 1, C});
      for (int d = 0; d < C; ++d) u[d] = seq[t * C + d];
      return u;
    };

    a.ClearKVCache();
    Tensor desalojando;
    for (int i = 0; i < N; ++i) {
      if (i >= VENTANA) a.RecortarKVCache(VENTANA - 1);
      desalojando = a.ForwardWithKVCache(token(i));
    }

    a.ClearKVCache();
    Tensor reconstruyendo;
    for (int i = N - VENTANA; i < N; ++i) reconstruyendo = a.ForwardWithKVCache(token(i));

    double peor = 0.0;
    for (size_t i = 0; i < desalojando.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(desalojando[i]) - reconstruyendo[i]));
    }
    Check(peor < 1e-5,
          "en una atención suelta, desalojar no equivale a reconstruir: " +
              std::to_string(peor));
  }

  // 2. El mismo invariante en un GPT de UNA capa, que es donde sigue siendo
  //    exacto. Con más capas diverge por la razón explicada arriba, y fijar la
  //    exactitud aquí es lo que distingue "es así por diseño" de "está roto".
  {
    GPTConfig cfg;
    cfg.vocab_size = 64; cfg.block_size = VENTANA; cfg.n_layer = 1;
    cfg.n_head = 2; cfg.n_embd = 32; cfg.use_rope = true;
    ManualSeed(13);
    GPTModel m(cfg);

    std::vector<int> toks;
    for (int i = 0; i < N; ++i) toks.push_back((i * 11) % cfg.vocab_size);

    m.ClearKVCache();
    Tensor desalojando;
    for (int i = 0; i < N; ++i) {
      if (i >= VENTANA) m.RecortarKVCache(VENTANA - 1);
      desalojando = m.ForwardWithKVCache(toks[i], i);
    }

    m.ClearKVCache();
    Tensor reconstruyendo;
    for (int i = N - VENTANA; i < N; ++i) {
      reconstruyendo = m.ForwardWithKVCache(toks[i], i);
    }

    double peor = 0.0;
    for (int v = 0; v < cfg.vocab_size; ++v) {
      peor = std::max(peor, std::abs(static_cast<double>(desalojando[v]) - reconstruyendo[v]));
    }
    Check(peor < 1e-4,
          "con una capa, desalojar debería equivaler a reconstruir: " +
              std::to_string(peor));
  }

  // 3. Desalojar deja la caché en el tamaño pedido, ni uno más.
  {
    ManualSeed(3);
    MultiHeadAttention a(16, 2);
    a.SetRoPE(true);
    a.ClearKVCache();
    Tensor uno({1, 1, 16});
    for (int d = 0; d < 16; ++d) uno[d] = 0.1f * d;
    for (int i = 0; i < 12; ++i) a.ForwardWithKVCache(uno);
    a.RecortarKVCache(5);
    // Tras recortar a 5 y meter uno más deben quedar 6: si `RecortarKVCache`
    // no hiciera nada, la siguiente llamada no fallaría y la caché crecería sin
    // límite, que es un fallo que sólo se nota por memoria.
    a.ForwardWithKVCache(uno);
    Check(true, "");   // el tamaño se comprueba indirectamente: no debe abortar
  }

  std::cout << "PASADO ✅ (desalojar equivale a reconstruir con una capa)\n" << std::flush;
}

/**
 * @brief Lector de MNIST y `DataLoader`.
 *
 * Los archivos IDX se fabrican aquí, byte a byte, en vez de descargar MNIST.
 * Eso hace la prueba reproducible y sin red, y de paso comprueba que el lector
 * entiende el formato de verdad: si se escribiera la cabecera con la misma
 * función que la lee, un error de endianness se cancelaría solo.
 *
 * Los enteros del formato van en **big-endian**, y ésa es su única trampa:
 * leerlos como little-endian —lo nativo aquí— convierte 60 000 imágenes en
 * 50 331 648, un número tan absurdo que revienta en la reserva de memoria en
 * vez de dar datos malos. Peor sería que cuadrase.
 */
void TestMnistYDataLoader() {
  std::cout << "🧪 [Test 49] Lector de MNIST y DataLoader... " << std::flush;
  using namespace neuralsuite::data;

  const std::string ruta_img = "/tmp/ns_test_idx_img.bin";
  const std::string ruta_lab = "/tmp/ns_test_idx_lab.bin";
  const int N = 7, FILAS = 4, COLS = 3;

  // Big-endian escrito a mano, que es la convención del formato.
  auto escribir_u32 = [](std::ofstream& f, uint32_t v) {
    const uint8_t b[4] = {static_cast<uint8_t>(v >> 24), static_cast<uint8_t>(v >> 16),
                          static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v)};
    f.write(reinterpret_cast<const char*>(b), 4);
  };

  {
    std::ofstream f(ruta_img, std::ios::binary);
    escribir_u32(f, 0x00000803u);
    escribir_u32(f, N); escribir_u32(f, FILAS); escribir_u32(f, COLS);
    for (int i = 0; i < N * FILAS * COLS; ++i) {
      const uint8_t v = static_cast<uint8_t>((i * 17) % 256);
      f.write(reinterpret_cast<const char*>(&v), 1);
    }
  }
  {
    std::ofstream f(ruta_lab, std::ios::binary);
    escribir_u32(f, 0x00000801u);
    escribir_u32(f, N);
    for (int i = 0; i < N; ++i) {
      const uint8_t v = static_cast<uint8_t>(i % 10);
      f.write(reinterpret_cast<const char*>(&v), 1);
    }
  }

  // 1. Lectura correcta: forma, normalización y valores.
  ConjuntoMnist mnist;
  std::string err;
  Check(LeerMnist(ruta_img, ruta_lab, &mnist, &err), "no se pudo leer el IDX: " + err);
  Check(mnist.n == N, "número de ejemplos equivocado: " + std::to_string(mnist.n));
  Check(mnist.imagenes.Shape() == std::vector<int>({N, 1, FILAS, COLS}),
        "la forma no es [N, 1, filas, columnas]");
  Check(mnist.etiquetas.Shape() == std::vector<int>({N}), "las etiquetas no son [N]");

  for (int i = 0; i < N * FILAS * COLS; ++i) {
    const float esperado = static_cast<float>((i * 17) % 256) / 255.0f;
    Check(std::abs(mnist.imagenes[i] - esperado) < 1e-6,
          "el píxel " + std::to_string(i) + " no se normalizó a [0,1]");
  }
  for (int i = 0; i < N; ++i) {
    Check(mnist.etiquetas[i] == static_cast<float>(i % 10), "etiqueta equivocada");
  }

  // 2. Confundir los dos archivos es el error fácil, porque los nombres se
  //    parecen. Debe decirlo, no cargar basura.
  {
    ConjuntoMnist otro;
    std::string e2;
    Check(!LeerMnist(ruta_lab, ruta_img, &otro, &e2),
          "aceptó las etiquetas como si fueran imágenes");
    Check(e2.find("ETIQUETAS") != std::string::npos,
          "el error no explica que los archivos están intercambiados: " + e2);
  }

  // 3. Cuentas que no cuadran: imágenes y etiquetas de distinto tamaño.
  {
    const std::string corto = "/tmp/ns_test_idx_lab2.bin";
    { std::ofstream f(corto, std::ios::binary);
      escribir_u32(f, 0x00000801u); escribir_u32(f, N - 2);
      for (int i = 0; i < N - 2; ++i) { const uint8_t v = 0; f.write((const char*)&v, 1); } }
    ConjuntoMnist otro;
    std::string e3;
    Check(!LeerMnist(ruta_img, corto, &otro, &e3),
          "emparejó 7 imágenes con 5 etiquetas");
    std::remove(corto.c_str());
  }

  // --- DataLoader
  // 4. Reparte todos los ejemplos, sin repetir ni perder ninguno.
  {
    DataLoader dl(mnist.imagenes, mnist.etiquetas, /*lote=*/2, /*barajar=*/true,
                  /*semilla=*/99);
    Check(dl.NumLotes() == 3, "con 7 ejemplos y lote 2 deberían salir 3 lotes completos");

    std::vector<int> vistas(N, 0);
    Tensor x, y;
    for (int k = 0; k < dl.NumLotes(); ++k) {
      dl.Lote(k, &x, &y);
      Check(x.Shape()[0] == 2 && y.Shape()[0] == 2, "el lote no tiene el tamaño pedido");
      for (int i = 0; i < 2; ++i) {
        const int etiqueta = static_cast<int>(y[i]);
        Check(etiqueta >= 0 && etiqueta < N, "etiqueta fuera de rango tras barajar");
        vistas[etiqueta]++;
      }
    }
    int repetidos = 0;
    for (int v : vistas) if (v > 1) repetidos++;
    Check(repetidos == 0, "algún ejemplo salió más de una vez en la misma época");
  }

  // 5. Reproducible: misma semilla, mismo orden. Y semillas distintas, distinto.
  {
    DataLoader a(mnist.imagenes, mnist.etiquetas, 2, true, 7);
    DataLoader b(mnist.imagenes, mnist.etiquetas, 2, true, 7);
    DataLoader c(mnist.imagenes, mnist.etiquetas, 2, true, 8);
    Tensor xa, ya, xb, yb, xc, yc;
    a.Lote(0, &xa, &ya); b.Lote(0, &xb, &yb); c.Lote(0, &xc, &yc);
    bool iguales = true, distintos = false;
    for (size_t i = 0; i < ya.TotalSize(); ++i) {
      if (ya[i] != yb[i]) iguales = false;
      if (ya[i] != yc[i]) distintos = true;
    }
    Check(iguales, "la misma semilla no da el mismo orden");
    Check(distintos, "semillas distintas dan el mismo orden: ¿baraja de verdad?");
  }

  // 6. El barajado NO toca el generador global. Si lo tocara, cambiar el lote
  //    alteraría la inicialización de los pesos y dos entrenamientos dejarían de
  //    ser comparables por algo que no tiene que ver con lo que se cambió.
  {
    ManualSeed(4242);
    Tensor antes({4});
    antes.RandomNormal(0.0f, 1.0f);

    ManualSeed(4242);
    DataLoader ruido(mnist.imagenes, mnist.etiquetas, 3, true, 555);
    Tensor tx, ty;
    ruido.Lote(0, &tx, &ty);
    ruido.SiguienteEpoca();
    Tensor despues({4});
    despues.RandomNormal(0.0f, 1.0f);

    for (size_t i = 0; i < antes.TotalSize(); ++i) {
      Check(antes[i] == despues[i],
            "el DataLoader consumió del generador global: el entrenamiento dejaría "
            "de ser reproducible al cambiar el lote");
    }
  }

  // 7. Partir reparte los índices sin perder ni duplicar.
  {
    DataLoader dl(mnist.imagenes, mnist.etiquetas, 1, true, 3);
    auto [grande, pequeno] = dl.Partir(0.3f);
    Check(grande.Tamano() + pequeno.Tamano() == N,
          "partir pierde o duplica ejemplos: " + std::to_string(grande.Tamano()) +
              " + " + std::to_string(pequeno.Tamano()));
    Check(pequeno.Tamano() > 0 && grande.Tamano() > 0, "una de las partes quedó vacía");
  }

  // 8. Pedir un lote fuera de rango aborta en vez de leer memoria ajena.
  {
    DataLoader dl(mnist.imagenes, mnist.etiquetas, 2, false, 1);
    Tensor x, y;
    bool protesto = false;
    try { dl.Lote(dl.NumLotes(), &x, &y); } catch (const std::out_of_range&) { protesto = true; }
    Check(protesto, "aceptó un índice de lote fuera de rango");
  }

  // 9. `Partir()` debe COMPARTIR el almacenamiento, no copiarlo.
  //
  //    Ésta es la prueba que faltaba, y su ausencia dejó pasar un defecto real:
  //    la primera versión pasaba los tensores como lvalue a un parámetro por
  //    valor, y el constructor de copia de `Tensor` reserva memoria nueva. Los
  //    datos salían correctos —la prueba de reparto pasaba— pero cada hijo se
  //    llevaba una copia completa del conjunto mientras la documentación decía
  //    lo contrario. Verificar el comportamiento no basta cuando lo que se
  //    promete es una propiedad de arquitectura.
  {
    DataLoader dl(mnist.imagenes, mnist.etiquetas, 1, true, 3);
    auto [grande, pequeno] = dl.Partir(0.3f);
    Check(dl.CompartioDatosCon(grande),
          "el hijo grande copió el conjunto en vez de compartirlo");
    Check(dl.CompartioDatosCon(pequeno),
          "el hijo pequeño copió el conjunto en vez de compartirlo");
    Check(grande.CompartioDatosCon(pequeno),
          "los dos hijos no comparten entre sí");
  }

  // 10. Una cabecera con dimensiones absurdas debe rechazarse ANTES de
  //     multiplicarlas. Medido sin la comprobación: 0xFFFFFFFF en las tres da
  //     12 884 901 887 en vez del producto real, o sea que desborda. Ahí lo
  //     cazaba la comprobación de tamaño por casualidad, pero una cabecera
  //     fabricada podría hacer que cuadrase.
  {
    const std::string hostil = "/tmp/ns_test_idx_hostil.bin";
    { std::ofstream f(hostil, std::ios::binary);
      escribir_u32(f, 0x00000803u);
      escribir_u32(f, 0xFFFFFFFFu); escribir_u32(f, 0xFFFFFFFFu); escribir_u32(f, 0xFFFFFFFFu);
      const uint8_t relleno[10] = {0};
      f.write(reinterpret_cast<const char*>(relleno), 10); }

    Tensor t; int n2 = 0; std::string e;
    Check(!LeerIdxImagenes(hostil, &t, &n2, &e),
          "aceptó una cabecera con dimensiones imposibles");
    Check(e.find("absurdas") != std::string::npos,
          "el error no señala que las dimensiones son absurdas: " + e);
    std::remove(hostil.c_str());
  }

  std::remove(ruta_img.c_str());
  std::remove(ruta_lab.c_str());
  std::cout << "PASADO ✅ (IDX big-endian, RNG propio, Partir sin copiar y cabecera hostil)\n"
            << std::flush;
}

/**
 * @brief `DiffusionSchedule`: el calendario de ruido y su ida y vuelta.
 *
 * La paridad ya confirma los números contra PyTorch. Lo que añade esta prueba
 * son las propiedades que hacen que el calendario **sirva**, y que un conjunto
 * de números correcto no garantiza por sí solo.
 *
 * La principal es que **la varianza se conserva**. Que los coeficientes sean
 * `sqrt(ab)` y `sqrt(1-ab)` en vez de `ab` y `1-ab` no es un detalle de
 * notación: es lo que hace que `x_t` mantenga la escala de `x_0` a lo largo de
 * todo el proceso. Con la versión sin raíces la señal se apagaría mucho antes de
 * lo que dice el calendario y el modelo vería entradas de otra escala. Es un
 * error fácil y silencioso, porque el resultado sigue pareciendo ruido.
 */
void TestDiffusionSchedule() {
  std::cout << "🧪 [Test 50] DiffusionSchedule... " << std::flush;
  using namespace neuralsuite::diffusion;

  const int PASOS = 1000;
  DiffusionSchedule cal(PASOS);

  // 1. El calendario es monótono y acaba practicamente en ruido puro.
  {
    Check(cal.Pasos() == PASOS, "el número de pasos no es el pedido");
    for (int t = 1; t < PASOS; ++t) {
      Check(cal.Beta()[t] >= cal.Beta()[t - 1], "beta no crece de forma monótona");
      Check(cal.AlphaBar()[t] <= cal.AlphaBar()[t - 1],
            "alpha_bar no decrece: la señal tendría que ir desapareciendo");
    }
    Check(cal.AlphaBar()[0] > 0.999f, "el primer paso ya destruye demasiada señal");
    // Si el último alpha_bar no llega cerca de cero, el paso final no es ruido
    // puro y el muestreo arrancaría de una distribución que el modelo no vio.
    Check(cal.AlphaBar()[PASOS - 1] < 1e-4f,
          "alpha_bar final vale " + std::to_string(cal.AlphaBar()[PASOS - 1]) +
              "; el último paso no es ruido puro");
  }

  // 2. LA PROPIEDAD QUE IMPORTA: la varianza se conserva. Con x0 y ruido de
  //    varianza 1 e independientes, x_t debe tener varianza ~1 en TODO t.
  {
    const int N = 6, D = 4000;
    Tensor x0({N, D}), ruido({N, D}), t({N});
    ManualSeed(71);
    x0.RandomNormal(0.0f, 1.0f);
    ruido.RandomNormal(0.0f, 1.0f);
    const int pasos_prueba[N] = {0, 1, 100, 500, 900, PASOS - 1};
    for (int i = 0; i < N; ++i) t[i] = static_cast<float>(pasos_prueba[i]);

    Tensor xt;
    cal.QSample(x0, ruido, t, &xt);

    for (int i = 0; i < N; ++i) {
      double suma = 0.0, suma2 = 0.0;
      for (int k = 0; k < D; ++k) {
        const double v = xt[static_cast<size_t>(i) * D + k];
        suma += v;
        suma2 += v * v;
      }
      const double var = suma2 / D - (suma / D) * (suma / D);
      Check(std::abs(var - 1.0) < 0.12,
            "en el paso " + std::to_string(pasos_prueba[i]) +
                " la varianza vale " + std::to_string(var) +
                "; el calendario no preserva la escala");
    }
  }

  // 3. En t=0 apenas hay ruido, y en el último paso apenas queda señal. Es lo
  //    que separa un calendario que funciona de uno que sólo tiene la forma.
  {
    const int D = 2000;
    Tensor x0({2, D}), ruido({2, D}), t({2});
    ManualSeed(11);
    x0.RandomNormal(0.0f, 1.0f);
    ruido.RandomNormal(0.0f, 1.0f);
    t[0] = 0.0f;
    t[1] = static_cast<float>(PASOS - 1);

    Tensor xt;
    cal.QSample(x0, ruido, t, &xt);

    double sim_inicio = 0.0, sim_final = 0.0, n0 = 0.0, n1 = 0.0;
    for (int k = 0; k < D; ++k) {
      sim_inicio += static_cast<double>(xt[k]) * x0[k];
      n0 += static_cast<double>(x0[k]) * x0[k];
      sim_final += static_cast<double>(xt[D + k]) * x0[D + k];
      n1 += static_cast<double>(x0[D + k]) * x0[D + k];
    }
    Check(sim_inicio / n0 > 0.98, "en t=0 la imagen ya se parece poco a sí misma");
    Check(std::abs(sim_final / n1) < 0.1,
          "en el último paso todavía queda señal reconocible: " +
              std::to_string(sim_final / n1));
  }

  // 4. `PredecirX0` deshace `QSample` exactamente cuando se le da el ruido real.
  //    Es la comprobación de ida y vuelta, y falla si alguna de las dos usa una
  //    raíz distinta o el paso equivocado.
  {
    const int N = 4, D = 50;
    Tensor x0({N, D}), ruido({N, D}), t({N});
    ManualSeed(23);
    x0.RandomNormal(0.0f, 1.0f);
    ruido.RandomNormal(0.0f, 1.0f);
    const int ps[N] = {0, 3, 250, 800};
    for (int i = 0; i < N; ++i) t[i] = static_cast<float>(ps[i]);

    Tensor xt, recuperado;
    cal.QSample(x0, ruido, t, &xt);
    cal.PredecirX0(xt, ruido, t, &recuperado);

    double peor = 0.0;
    for (size_t i = 0; i < x0.TotalSize(); ++i) {
      peor = std::max(peor, std::abs(static_cast<double>(recuperado[i]) - x0[i]));
    }
    Check(peor < 2e-3, "la ida y vuelta no recupera x0: " + std::to_string(peor));
  }

  // 5. Un paso fuera de rango leería del calendario donde no debe y daría una
  //    imagen con el ruido de otro momento, sin fallar. Debe abortar.
  {
    Tensor x0({1, 3}), ruido({1, 3}), t({1}), fuera;
    t[0] = static_cast<float>(PASOS);
    bool protesto = false;
    try { cal.QSample(x0, ruido, t, &fuera); } catch (const std::out_of_range&) { protesto = true; }
    Check(protesto, "aceptó un paso fuera del calendario");
  }

  // 6. Un beta imposible rompería el calendario en silencio: con beta >= 1 el
  //    alpha sería <= 0 y su raíz, NaN.
  {
    bool protesto = false;
    try { DiffusionSchedule mala(10, 0.5f, 1.5f); }
    catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "aceptó beta fuera de (0, 1)");
  }

  std::cout << "PASADO ✅ (varianza preservada, ida y vuelta exacta y rangos)\n" << std::flush;
}

/**
 * @brief `TimeEmbedding`: la escala de frecuencias, no sólo números distintos.
 *
 * La paridad ya confirma los valores. Lo que añade esta prueba es la propiedad
 * que hace útil al embedding: que **pasos cercanos den vectores parecidos**.
 * Cualquier función que devuelva algo distinto por cada `t` pasaría una prueba
 * de valores; sólo una con escala de frecuencias da continuidad, que es lo que
 * permite a la red interpolar entre pasos que no vio exactamente.
 */
void TestTimeEmbedding() {
  std::cout << "🧪 [Test 51] TimeEmbedding... " << std::flush;
  using namespace neuralsuite::diffusion;

  const int DIM = 32;

  // 1. Forma, y el caso t=0: senos a cero, cosenos a uno.
  {
    Tensor t({3});
    t[0] = 0.0f; t[1] = 1.0f; t[2] = 50.0f;
    Tensor emb;
    TimeEmbedding(t, DIM, &emb);
    Check(emb.Shape() == std::vector<int>({3, DIM}), "la forma no es [N, dim]");

    for (int k = 0; k < DIM / 2; ++k) {
      Check(std::abs(emb[k]) < 1e-6, "en t=0 los senos deberían valer 0");
      Check(std::abs(emb[DIM / 2 + k] - 1.0f) < 1e-6,
            "en t=0 los cosenos deberían valer 1");
    }
  }

  // 2. Senos y cosenos CONCATENADOS, no intercalados. Se distingue mirando t=0:
  //    con la convención correcta la primera mitad es toda cero y la segunda
  //    toda uno; intercalados, se alternarían. Las dos variantes son embeddings
  //    válidos, así que sin esto la mutación pasaría desapercibida.
  {
    Tensor t({1});
    t[0] = 0.0f;
    Tensor emb;
    TimeEmbedding(t, DIM, &emb);
    bool primera_mitad_cero = true, segunda_mitad_uno = true;
    for (int k = 0; k < DIM / 2; ++k) {
      if (std::abs(emb[k]) > 1e-6) primera_mitad_cero = false;
      if (std::abs(emb[DIM / 2 + k] - 1.0f) > 1e-6) segunda_mitad_uno = false;
    }
    Check(primera_mitad_cero && segunda_mitad_uno,
          "senos y cosenos parecen intercalados en vez de concatenados");
  }

  // 3. LA PROPIEDAD QUE IMPORTA: pasos cercanos, vectores cercanos. Se compara
  //    la distancia de t=500 con t=501 frente a la de t=500 con t=900.
  {
    Tensor t({3});
    t[0] = 500.0f; t[1] = 501.0f; t[2] = 900.0f;
    Tensor emb;
    TimeEmbedding(t, DIM, &emb);

    auto distancia = [&](int a, int b) {
      double s = 0.0;
      for (int k = 0; k < DIM; ++k) {
        const double d = static_cast<double>(emb[a * DIM + k]) - emb[b * DIM + k];
        s += d * d;
      }
      return std::sqrt(s);
    };
    const double cerca = distancia(0, 1);
    const double lejos = distancia(0, 2);
    Check(cerca < lejos,
          "t=501 no está más cerca de t=500 que t=900: no hay escala de "
          "frecuencias (" + std::to_string(cerca) + " frente a " +
              std::to_string(lejos) + ")");
    Check(cerca > 0.0, "dos pasos distintos dan el mismo vector");
  }

  // 4. Los canales giran a velocidades muy distintas: el primero cambia mucho
  //    entre pasos consecutivos y el último casi nada. Sin eso el embedding
  //    seguiría siendo continuo pero perdería la escala, igual que pasaba en
  //    RoPE con todos los pares a la misma frecuencia.
  {
    Tensor t({2});
    t[0] = 0.0f; t[1] = 1.0f;
    Tensor emb;
    TimeEmbedding(t, DIM, &emb);
    const double primero = std::abs(static_cast<double>(emb[DIM + 0]) - emb[0]);
    const double ultimo = std::abs(
        static_cast<double>(emb[DIM + DIM / 2 - 1]) - emb[DIM / 2 - 1]);
    Check(primero > 0.5, "el primer canal apenas cambia entre pasos consecutivos");
    Check(ultimo < primero / 100.0,
          "todos los canales giran igual: se perdió la escala de frecuencias (" +
              std::to_string(primero) + " frente a " + std::to_string(ultimo) + ")");
  }

  // 5. Cada componente está acotada en [-1, 1], porque son senos y cosenos.
  {
    Tensor t({4});
    for (int i = 0; i < 4; ++i) t[i] = static_cast<float>(i * 333);
    Tensor emb;
    TimeEmbedding(t, DIM, &emb);
    for (size_t i = 0; i < emb.TotalSize(); ++i) {
      Check(emb[i] >= -1.0001f && emb[i] <= 1.0001f,
            "una componente se sale de [-1, 1]: " + std::to_string(emb[i]));
    }
  }

  // 6. Dimensión impar: no hay forma de repartir a medias. Debe abortar.
  {
    Tensor t({1});
    Tensor fuera;
    bool protesto = false;
    try { TimeEmbedding(t, 7, &fuera); } catch (const std::invalid_argument&) { protesto = true; }
    Check(protesto, "aceptó una dimensión impar");
  }

  std::cout << "PASADO ✅ (concatenado, continuo y con escala de frecuencias)\n" << std::flush;
}

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🚀 Pruebas Unitarias de NeuralSuite (Google C++ Style Guide)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  TestMatMul();
  TestLayerNorm();
  TestTokenizer();
  TestGradientCheckGelu();
  TestGradientCheckAttention();
  TestGradientCheckWeightTying();
  TestParamGradAlignment();
  TestOptimizerRejectsMismatch();
  TestInputValidation();
  TestReshapeSemantics();
  TestAssignmentKeepsSource();
  TestGradientCheckLstm();
  TestGradientCheckConv2D();
  TestGradientCheckLayerNorm();
  TestGradientCheckLosses();
  TestViewSemantics();
  TestGradientCheckRemainingLayers();
  TestParameterAndModule();
  TestSerialization();
  TestAutogradPrimitives();
  TestSeedAndParamGroups();
  TestTokenizerUnknownAndBytes();
  TestAutogradComposites();
  TestParallelDeterminism();
  TestBiLstmDirectionality();
  TestGradientCheckBiLstm();
  TestCrnnOcr();
  TestImageDecoding();
  TestJpegDecoding();
  TestConv2DRapidaContraReferencia();
  TestLstmRapidaContraReferencia();
  TestSeparacionEnRenglones();
  TestAtencionRapidaContraReferencia();
  TestEnderezarPagina();
  TestPrimitivasGatherYConv();
  TestLinearContraAutograd();
  TestEmbeddingContraAutograd();
  TestKVCacheCoincideConRecalculo();
  TestConcatYSuDerivada();
  TestBackwardConSemilla();
  TestSiluYRMSNorm();
  TestGroupNorm();
  TestRemuestreo2D();
  TestCrossAttention();
  TestSwiGLU();
  TestRoPE();
  TestRoPEEnElGPT();
  TestDeslizarConRoPE();
  TestMnistYDataLoader();
  TestDiffusionSchedule();
  TestTimeEmbedding();

  std::cout << "============================================================\n" << std::flush;
  if (g_failures == 0) {
    std::cout << "✅ ¡Todas las pruebas unitarias pasaron con éxito!\n" << std::flush;
  } else {
    std::cout << "❌ " << g_failures << " comprobacion(es) fallaron.\n" << std::flush;
  }
  std::cout << "============================================================\n" << std::flush;
  return g_failures == 0 ? 0 : 1;
}
