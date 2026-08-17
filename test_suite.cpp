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
#include <functional>
#include <iostream>
#include <string>
#include <vector>
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

  std::cout << "============================================================\n" << std::flush;
  if (g_failures == 0) {
    std::cout << "✅ ¡Todas las pruebas unitarias pasaron con éxito!\n" << std::flush;
  } else {
    std::cout << "❌ " << g_failures << " comprobacion(es) fallaron.\n" << std::flush;
  }
  std::cout << "============================================================\n" << std::flush;
  return g_failures == 0 ? 0 : 1;
}
