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

  std::cout << "============================================================\n" << std::flush;
  if (g_failures == 0) {
    std::cout << "✅ ¡Todas las pruebas unitarias pasaron con éxito!\n" << std::flush;
  } else {
    std::cout << "❌ " << g_failures << " comprobacion(es) fallaron.\n" << std::flush;
  }
  std::cout << "============================================================\n" << std::flush;
  return g_failures == 0 ? 0 : 1;
}
