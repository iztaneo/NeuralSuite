// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file autograd.h
 * @brief Motor de diferenciacion automatica en modo inverso.
 *
 * Hasta ahora cada capa implementa su `Backward()` a mano. Funciona mientras el
 * catalogo sea pequeno, pero cada derivada escrita a mano es una oportunidad de
 * equivocarse, y los dos defectos que invalidaron el entrenamiento del
 * Transformer estaban precisamente ahi.
 *
 * Con autograd la derivada se deduce de la operacion. Cada operacion registra
 * como propagar el gradiente hacia sus entradas, y `Backward()` recorre el
 * grafo en orden topologico inverso aplicando la regla de la cadena.
 *
 *     auto x = Variable::Create(tensor, true);   // requires_grad
 *     auto y = Tanh(MatMul(x, w) + b);
 *     Backward(Sum(y));
 *     x->Grad();   // ya calculado, sin escribir ninguna derivada
 *
 * Limitaciones deliberadas de esta primera version, para no acumular
 * complejidad antes de tener las primitivas verificadas:
 *
 * - Solo `float32`.
 * - El grafo se construye siempre; no hay todavia un modo de inferencia que lo
 *   omita.
 *
 * **Este archivo se queda en la cabecera**: tiene plantillas, y ademas es un
 * motor de grafo donde cada primitiva se define junto a su derivada. Separarlas
 * pondria la funcion en un archivo y su gradiente en otro, que es justo el par
 * que hay que leer junto.
 */

#ifndef NEURAL_SUITE_INCLUDE_AUTOGRAD_H_
#define NEURAL_SUITE_INCLUDE_AUTOGRAD_H_

#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "layers/conv2d.h"
#include "tensor.h"

namespace neuralsuite {
namespace autograd {

class Variable;
using VarPtr = std::shared_ptr<Variable>;

/**
 * @class Variable
 * @brief Un tensor dentro del grafo, con su gradiente y como propagarlo.
 *
 * Los nodos apuntan a sus entradas y nunca al reves, de modo que el grafo es
 * aciclico y la memoria se libera sola cuando se suelta el resultado.
 */
class Variable : public std::enable_shared_from_this<Variable> {
 public:
  /** @brief Crea una hoja del grafo a partir de un tensor. */
  static VarPtr Create(Tensor value, bool requires_grad = false) {
    auto v = std::make_shared<Variable>(Private{});
    v->value_ = std::move(value);
    v->requires_grad_ = requires_grad;
    if (requires_grad) {
      v->grad_ = Tensor(v->value_.Shape());
      v->grad_.Zeros();
    }
    return v;
  }

  [[nodiscard]] const Tensor& Value() const { return value_; }
  [[nodiscard]] Tensor& Value() { return value_; }
  [[nodiscard]] const Tensor& Grad() const { return grad_; }
  [[nodiscard]] Tensor& Grad() { return grad_; }
  [[nodiscard]] bool RequiresGrad() const { return requires_grad_; }
  [[nodiscard]] const std::vector<int>& Shape() const { return value_.Shape(); }

  void ZeroGrad() {
    if (grad_.TotalSize() > 0) grad_.Zeros();
  }

  /** @brief Suma `delta` al gradiente acumulado, creandolo si hace falta. */
  void AccumulateGrad(const Tensor& delta) {
    if (grad_.TotalSize() == 0) {
      grad_ = Tensor(value_.Shape());
      grad_.Zeros();
    }
    // Un mismo nodo puede alimentar varias operaciones; sus contribuciones se
    // suman, que es lo que dice la regla de la cadena para caminos multiples.
    for (size_t i = 0; i < grad_.TotalSize(); ++i) grad_[i] += delta[i];
  }

 private:
  struct Private {};

 public:
  explicit Variable(Private) {}

 private:
  friend VarPtr MakeOp(Tensor, std::vector<VarPtr>, std::function<void(const Tensor&)>);
  friend void Backward(const VarPtr&);

  Tensor value_;
  Tensor grad_;
  bool requires_grad_ = false;

  // Entradas de la operacion que produjo este nodo, y como repartir entre ellas
  // el gradiente que llegue.
  std::vector<VarPtr> parents_;
  std::function<void(const Tensor& grad_output)> backward_;
};

/**
 * @brief Construye el nodo resultado de una operacion.
 *
 * `backward` recibe el gradiente que llega al resultado y debe repartirlo entre
 * las entradas con `AccumulateGrad`.
 */
inline VarPtr MakeOp(Tensor value, std::vector<VarPtr> parents,
                     std::function<void(const Tensor&)> backward) {
  auto v = std::make_shared<Variable>(Variable::Private{});
  v->value_ = std::move(value);
  // El resultado necesita gradiente si alguna de sus entradas lo necesita.
  v->requires_grad_ = std::any_of(parents.begin(), parents.end(),
                                  [](const VarPtr& p) { return p->RequiresGrad(); });
  v->parents_ = std::move(parents);
  v->backward_ = std::move(backward);
  return v;
}

/**
 * @brief Propaga el gradiente desde `root` hacia todas sus entradas.
 *
 * `root` debe ser escalar: el gradiente de una funcion respecto de si misma es
 * uno, y eso solo esta definido para un unico valor. Para una salida con varios
 * elementos hay que reducirla antes con `Sum` o `Mean`.
 *
 * El recorrido es en orden topologico inverso: un nodo no propaga hasta haber
 * recibido las contribuciones de todos los caminos que llegan a el, que es lo
 * que distingue un grafo de un simple encadenado lineal.
 */
inline void Backward(const VarPtr& root) {
  if (root->Value().TotalSize() != 1) {
    throw std::invalid_argument(
        "Backward: la raiz debe ser escalar; reduce la salida con Sum() o Mean(). "
        "Tiene " + std::to_string(root->Value().TotalSize()) + " elementos.");
  }

  // Orden topologico por recorrido en profundidad.
  std::vector<Variable*> order;
  std::unordered_set<Variable*> visited;
  std::function<void(const VarPtr&)> visit = [&](const VarPtr& node) {
    if (node == nullptr || visited.count(node.get())) return;
    visited.insert(node.get());
    for (const VarPtr& parent : node->parents_) visit(parent);
    order.push_back(node.get());
  };
  visit(root);

  Tensor seed({1});
  seed[0] = 1.0f;
  root->AccumulateGrad(seed);

  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    Variable* node = *it;
    if (node->backward_ && node->Grad().TotalSize() > 0) {
      node->backward_(node->Grad());
    }
  }
}

namespace detail {

/**
 * @brief Forma resultante de combinar dos operandos, alineando por la derecha.
 *
 * Cada eje debe coincidir o valer 1, en cuyo caso se repite. Es la regla
 * habitual: sumar un sesgo `[D]` a un lote `[N, D]` no deberia obligar a
 * materializar N copias del sesgo.
 */
inline std::vector<int> BroadcastShape(const std::vector<int>& a, const std::vector<int>& b,
                                       const char* op) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    // Se recorren de derecha a izquierda; los ejes que faltan valen 1.
    const int da = (i < a.size()) ? a[a.size() - 1 - i] : 1;
    const int db = (i < b.size()) ? b[b.size() - 1 - i] : 1;
    if (da != db && da != 1 && db != 1) {
      throw std::invalid_argument(std::string(op) + ": formas incompatibles para broadcasting.");
    }
    out[rank - 1 - i] = std::max(da, db);
  }
  return out;
}

/** @brief Pasos por eje de un tensor contiguo, con 0 donde el eje se repite. */
inline std::vector<size_t> BroadcastStrides(const std::vector<int>& shape,
                                            const std::vector<int>& out_shape) {
  std::vector<size_t> strides(out_shape.size(), 0);
  size_t running = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    const size_t axis_out = out_shape.size() - 1 - i;
    const int dim = (i < shape.size()) ? shape[shape.size() - 1 - i] : 1;
    // Paso cero significa "este eje se repite": todos los indices de salida
    // leen el mismo elemento de la entrada.
    strides[axis_out] = (dim == 1) ? 0 : running;
    running *= static_cast<size_t>(dim);
  }
  return strides;
}

/** @brief Recorre la forma de salida aplicando `fn(idx_salida, idx_a, idx_b)`. */
template <typename Fn>
void ForEachBroadcast(const std::vector<int>& out_shape, const std::vector<size_t>& sa,
                      const std::vector<size_t>& sb, Fn&& fn) {
  size_t total = 1;
  for (int d : out_shape) total *= static_cast<size_t>(d);

  std::vector<int> counter(out_shape.size(), 0);
  size_t ia = 0, ib = 0;
  for (size_t i = 0; i < total; ++i) {
    fn(i, ia, ib);
    // Incremento posicional del contador multidimensional.
    for (size_t axis = out_shape.size(); axis-- > 0;) {
      ia += sa[axis];
      ib += sb[axis];
      if (++counter[axis] < out_shape[axis]) break;
      ia -= sa[axis] * static_cast<size_t>(out_shape[axis]);
      ib -= sb[axis] * static_cast<size_t>(out_shape[axis]);
      counter[axis] = 0;
    }
  }
}

/**
 * @brief Reduce un gradiente con la forma de salida a la forma del operando.
 *
 * Un eje que se repitio en el forward recibe, en el backward, la suma de todas
 * las posiciones que lo usaron: si un valor influyo en N salidas, su gradiente
 * es la suma de las N contribuciones.
 */
inline Tensor ReduceToShape(const Tensor& grad, const std::vector<int>& out_shape,
                            const std::vector<int>& target_shape) {
  Tensor reduced(target_shape);
  reduced.Zeros();
  const std::vector<size_t> strides = BroadcastStrides(target_shape, out_shape);
  const std::vector<size_t> zero(out_shape.size(), 0);
  ForEachBroadcast(out_shape, strides, zero,
                   [&](size_t i, size_t it, size_t) { reduced[it] += grad[i]; });
  return reduced;
}

}  // namespace detail

// ============================================================================
// PRIMITIVAS
// ============================================================================

/** @brief Suma elemento a elemento, con broadcasting. */
inline VarPtr Add(const VarPtr& a, const VarPtr& b) {
  const std::vector<int> shape = detail::BroadcastShape(a->Shape(), b->Shape(), "Add");
  const auto sa = detail::BroadcastStrides(a->Shape(), shape);
  const auto sb = detail::BroadcastStrides(b->Shape(), shape);

  Tensor out(shape);
  detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
    out[i] = a->Value()[ia] + b->Value()[ib];
  });

  return MakeOp(std::move(out), {a, b}, [a, b, shape](const Tensor& g) {
    if (a->RequiresGrad()) a->AccumulateGrad(detail::ReduceToShape(g, shape, a->Shape()));
    if (b->RequiresGrad()) b->AccumulateGrad(detail::ReduceToShape(g, shape, b->Shape()));
  });
}

/** @brief Producto elemento a elemento, con broadcasting. */
inline VarPtr Mul(const VarPtr& a, const VarPtr& b) {
  const std::vector<int> shape = detail::BroadcastShape(a->Shape(), b->Shape(), "Mul");
  const auto sa = detail::BroadcastStrides(a->Shape(), shape);
  const auto sb = detail::BroadcastStrides(b->Shape(), shape);

  Tensor out(shape);
  detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
    out[i] = a->Value()[ia] * b->Value()[ib];
  });

  return MakeOp(std::move(out), {a, b}, [a, b, shape, sa, sb](const Tensor& g) {
    Tensor ga(shape), gb(shape);
    detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
      ga[i] = g[i] * b->Value()[ib];
      gb[i] = g[i] * a->Value()[ia];
    });
    if (a->RequiresGrad()) a->AccumulateGrad(detail::ReduceToShape(ga, shape, a->Shape()));
    if (b->RequiresGrad()) b->AccumulateGrad(detail::ReduceToShape(gb, shape, b->Shape()));
  });
}

/** @brief Resta elemento a elemento, con broadcasting. */
inline VarPtr Sub(const VarPtr& a, const VarPtr& b) {
  const std::vector<int> shape = detail::BroadcastShape(a->Shape(), b->Shape(), "Sub");
  const auto sa = detail::BroadcastStrides(a->Shape(), shape);
  const auto sb = detail::BroadcastStrides(b->Shape(), shape);

  Tensor out(shape);
  detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
    out[i] = a->Value()[ia] - b->Value()[ib];
  });

  return MakeOp(std::move(out), {a, b}, [a, b, shape](const Tensor& g) {
    if (a->RequiresGrad()) a->AccumulateGrad(detail::ReduceToShape(g, shape, a->Shape()));
    if (b->RequiresGrad()) {
      Tensor neg(shape);
      for (size_t i = 0; i < neg.TotalSize(); ++i) neg[i] = -g[i];
      b->AccumulateGrad(detail::ReduceToShape(neg, shape, b->Shape()));
    }
  });
}

/** @brief Producto de matrices. dC/dA = G·Bᵀ, dC/dB = Aᵀ·G. */
inline VarPtr MatMulVar(const VarPtr& a, const VarPtr& b) {
  Tensor out;
  MatMul(a->Value(), b->Value(), out);

  return MakeOp(std::move(out), {a, b}, [a, b](const Tensor& g) {
    if (a->RequiresGrad()) {
      Tensor bt = Transpose(b->Value());
      Tensor da;
      MatMul(g, bt, da);
      a->AccumulateGrad(da);
    }
    if (b->RequiresGrad()) {
      Tensor at = Transpose(a->Value());
      Tensor db;
      MatMul(at, g, db);
      b->AccumulateGrad(db);
    }
  });
}

/** @brief Suma de todos los elementos. Cada uno contribuye con 1. */
inline VarPtr Sum(const VarPtr& a) {
  Tensor out({1});
  double acc = 0.0;
  for (size_t i = 0; i < a->Value().TotalSize(); ++i) acc += a->Value()[i];
  out[0] = static_cast<float>(acc);

  return MakeOp(std::move(out), {a}, [a](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = g[0];
    a->AccumulateGrad(da);
  });
}

/** @brief Media de todos los elementos. Cada uno contribuye con 1/N. */
inline VarPtr Mean(const VarPtr& a) {
  const size_t n = a->Value().TotalSize();
  Tensor out({1});
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) acc += a->Value()[i];
  out[0] = static_cast<float>(acc / static_cast<double>(n));

  return MakeOp(std::move(out), {a}, [a, n](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    const float share = g[0] / static_cast<float>(n);
    for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = share;
    a->AccumulateGrad(da);
  });
}

/** @brief Exponencial. d(e^x)/dx = e^x, que ya es la salida. */
inline VarPtr Exp(const VarPtr& a) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = std::exp(a->Value()[i]);

  Tensor cached = out;
  return MakeOp(std::move(out), {a}, [a, cached](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = g[i] * cached[i];
    a->AccumulateGrad(da);
  });
}

/** @brief Logaritmo natural. d(ln x)/dx = 1/x. */
inline VarPtr Log(const VarPtr& a) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = std::log(a->Value()[i]);

  return MakeOp(std::move(out), {a}, [a](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = g[i] / a->Value()[i];
    a->AccumulateGrad(da);
  });
}

/** @brief Tangente hiperbolica. d(tanh x)/dx = 1 - tanh²x. */
inline VarPtr Tanh(const VarPtr& a) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = std::tanh(a->Value()[i]);

  Tensor cached = out;
  return MakeOp(std::move(out), {a}, [a, cached](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) {
      da[i] = g[i] * (1.0f - cached[i] * cached[i]);
    }
    a->AccumulateGrad(da);
  });
}

/** @brief Relu. La derivada en cero se toma como cero. */
inline VarPtr Relu(const VarPtr& a) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) {
    out[i] = a->Value()[i] > 0.0f ? a->Value()[i] : 0.0f;
  }

  return MakeOp(std::move(out), {a}, [a](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) {
      da[i] = a->Value()[i] > 0.0f ? g[i] : 0.0f;
    }
    a->AccumulateGrad(da);
  });
}

/** @brief Reinterpreta los ejes. El gradiente vuelve con la forma original. */
inline VarPtr Reshape(const VarPtr& a, const std::vector<int>& shape) {
  Tensor out = a->Value();
  out.Reshape(shape);

  const std::vector<int> original = a->Shape();
  return MakeOp(std::move(out), {a}, [a, original](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da = g;
    da.Reshape(original);
    a->AccumulateGrad(da);
  });
}

/** @brief Transpuesta de una matriz. El gradiente se transpone de vuelta. */
inline VarPtr TransposeVar(const VarPtr& a) {
  Tensor out = Transpose(a->Value());

  return MakeOp(std::move(out), {a}, [a](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    a->AccumulateGrad(Transpose(g));
  });
}


/** @brief Division elemento a elemento, con broadcasting. */
inline VarPtr Div(const VarPtr& a, const VarPtr& b) {
  const std::vector<int> shape = detail::BroadcastShape(a->Shape(), b->Shape(), "Div");
  const auto sa = detail::BroadcastStrides(a->Shape(), shape);
  const auto sb = detail::BroadcastStrides(b->Shape(), shape);

  Tensor out(shape);
  detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
    out[i] = a->Value()[ia] / b->Value()[ib];
  });

  return MakeOp(std::move(out), {a, b}, [a, b, shape, sa, sb](const Tensor& g) {
    Tensor ga(shape), gb(shape);
    detail::ForEachBroadcast(shape, sa, sb, [&](size_t i, size_t ia, size_t ib) {
      const float bv = b->Value()[ib];
      ga[i] = g[i] / bv;                                    // d(a/b)/da = 1/b
      gb[i] = -g[i] * a->Value()[ia] / (bv * bv);           // d(a/b)/db = -a/b^2
    });
    if (a->RequiresGrad()) a->AccumulateGrad(detail::ReduceToShape(ga, shape, a->Shape()));
    if (b->RequiresGrad()) b->AccumulateGrad(detail::ReduceToShape(gb, shape, b->Shape()));
  });
}

/** @brief Raiz cuadrada. d(sqrt x)/dx = 1/(2 sqrt x). */
inline VarPtr Sqrt(const VarPtr& a) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = std::sqrt(a->Value()[i]);

  Tensor cached = out;
  return MakeOp(std::move(out), {a}, [a, cached](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = g[i] / (2.0f * cached[i]);
    a->AccumulateGrad(da);
  });
}

/** @brief Suma un escalar constante, sin gradiente propio. */
inline VarPtr AddScalar(const VarPtr& a, float value) {
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = a->Value()[i] + value;
  return MakeOp(std::move(out), {a}, [a](const Tensor& g) {
    if (a->RequiresGrad()) a->AccumulateGrad(g);
  });
}

namespace detail {

/** @brief Numero de elementos y paso del ultimo eje. */
inline void LastAxisLayout(const std::vector<int>& shape, size_t* rows, size_t* width) {
  if (shape.empty()) throw std::invalid_argument("Reduccion: el tensor no tiene ejes.");
  *width = static_cast<size_t>(shape.back());
  size_t total = 1;
  for (int d : shape) total *= static_cast<size_t>(d);
  *rows = total / *width;
}

}  // namespace detail

/**
 * @brief Suma a lo largo del ultimo eje, conservando el rango.
 *
 * `[N, D]` se reduce a `[N, 1]`, que con broadcasting vuelve a combinarse con
 * el original. Es lo que hace falta para expresar medias y normalizaciones sin
 * escribir sus derivadas.
 */
inline VarPtr SumLastAxis(const VarPtr& a) {
  size_t rows = 0, width = 0;
  detail::LastAxisLayout(a->Shape(), &rows, &width);

  std::vector<int> out_shape = a->Shape();
  out_shape.back() = 1;

  Tensor out(out_shape);
  for (size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (size_t c = 0; c < width; ++c) acc += a->Value()[r * width + c];
    out[r] = static_cast<float>(acc);
  }

  return MakeOp(std::move(out), {a}, [a, rows, width](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    // Cada elemento contribuyo con 1 a la suma de su fila.
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < width; ++c) da[r * width + c] = g[r];
    }
    a->AccumulateGrad(da);
  });
}

/** @brief Media a lo largo del ultimo eje, conservando el rango. */
inline VarPtr MeanLastAxis(const VarPtr& a) {
  size_t rows = 0, width = 0;
  detail::LastAxisLayout(a->Shape(), &rows, &width);

  std::vector<int> out_shape = a->Shape();
  out_shape.back() = 1;

  Tensor out(out_shape);
  for (size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (size_t c = 0; c < width; ++c) acc += a->Value()[r * width + c];
    out[r] = static_cast<float>(acc / static_cast<double>(width));
  }

  return MakeOp(std::move(out), {a}, [a, rows, width](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t r = 0; r < rows; ++r) {
      const float share = g[r] / static_cast<float>(width);
      for (size_t c = 0; c < width; ++c) da[r * width + c] = share;
    }
    a->AccumulateGrad(da);
  });
}

/**
 * @brief Softmax sobre el ultimo eje.
 *
 * Se implementa como primitiva y no por composicion porque la estabilidad
 * numerica exige restar el maximo de cada fila antes de exponenciar; expresarlo
 * con primitivas obligaria a derivar tambien por ese maximo, que no aporta nada
 * y complica el grafo.
 *
 * dx = p ⊙ (g − Σ(g ⊙ p)), con p la propia salida.
 */
inline VarPtr Softmax(const VarPtr& a) {
  size_t rows = 0, width = 0;
  detail::LastAxisLayout(a->Shape(), &rows, &width);

  Tensor out(a->Shape());
  for (size_t r = 0; r < rows; ++r) {
    float max_val = a->Value()[r * width];
    for (size_t c = 1; c < width; ++c) {
      max_val = std::max(max_val, a->Value()[r * width + c]);
    }
    double sum = 0.0;
    for (size_t c = 0; c < width; ++c) {
      const float e = std::exp(a->Value()[r * width + c] - max_val);
      out[r * width + c] = e;
      sum += e;
    }
    for (size_t c = 0; c < width; ++c) {
      out[r * width + c] /= static_cast<float>(sum);
    }
  }

  Tensor probs = out;
  return MakeOp(std::move(out), {a}, [a, probs, rows, width](const Tensor& g) {
    if (!a->RequiresGrad()) return;
    Tensor da(a->Shape());
    for (size_t r = 0; r < rows; ++r) {
      double dot = 0.0;
      for (size_t c = 0; c < width; ++c) dot += g[r * width + c] * probs[r * width + c];
      for (size_t c = 0; c < width; ++c) {
        da[r * width + c] = probs[r * width + c] * (g[r * width + c] - static_cast<float>(dot));
      }
    }
    a->AccumulateGrad(da);
  });
}

/**
 * @brief Toma filas de una tabla por indice, que es lo que hace un embedding.
 *
 * `tabla` es `[vocabulario, dimension]` e `indices` un tensor de enteros
 * guardados como float, con la forma que se quiera; la salida anade la
 * dimension del embedding al final. Es la operacion que `Embedding` hace a
 * mano.
 *
 * Hacia atras, cada fila recibe la suma de los gradientes de todas las
 * posiciones que la usaron. Esa suma es la razon de que no se pueda escribir
 * como una copia: en una secuencia el mismo token aparece muchas veces, y
 * quedarse con el ultimo gradiente en vez de sumarlos da un resultado que
 * parece razonable y esta mal.
 */
inline VarPtr Gather(const VarPtr& tabla, const Tensor& indices) {
  const std::vector<int>& forma_tabla = tabla->Shape();
  if (forma_tabla.size() != 2) {
    throw std::invalid_argument("Gather: la tabla debe ser [filas, dimension].");
  }
  const int filas = forma_tabla[0], dim = forma_tabla[1];

  std::vector<int> forma_salida = indices.Shape();
  forma_salida.push_back(dim);
  Tensor out(forma_salida);

  const size_t n = indices.TotalSize();
  for (size_t i = 0; i < n; ++i) {
    const int fila = static_cast<int>(indices[i]);
    if (fila < 0 || fila >= filas) {
      throw std::out_of_range("Gather: indice " + std::to_string(fila) +
                              " fuera de una tabla de " + std::to_string(filas) + " filas.");
    }
    std::memcpy(out.Data() + i * dim, tabla->Value().Data() + static_cast<size_t>(fila) * dim,
                dim * sizeof(float));
  }

  // Copia de los indices: la lambda vive mas que la llamada.
  Tensor guardados = indices;
  return MakeOp(std::move(out), {tabla}, [tabla, guardados, dim, n](const Tensor& g) {
    if (!tabla->RequiresGrad()) return;
    Tensor dtabla(tabla->Shape());
    dtabla.Zeros();
    for (size_t i = 0; i < n; ++i) {
      const size_t fila = static_cast<size_t>(guardados[i]);
      for (int d = 0; d < dim; ++d) dtabla[fila * dim + d] += g[i * dim + d];
    }
    tabla->AccumulateGrad(dtabla);
  });
}

/**
 * @brief Convolucion 2D como primitiva derivable.
 *
 * `entrada` es `[lote, canales, alto, ancho]` y `pesos`
 * `[canales_salida, canales_entrada, k, k]`. El sesgo va aparte porque sumarlo
 * es una operacion que el motor ya sabe derivar.
 *
 * Por dentro reutiliza `Conv2D`, que es la version rapida y esta comprobada
 * contra `Conv2DReference` y contra PyTorch. Envolver lo que ya funciona, en
 * vez de reescribir la convolucion con primitivas mas pequenas, tiene dos
 * ventajas: no se pierde el rendimiento —una convolucion compuesta de
 * multiplicaciones y sumas elementales seria mucho mas lenta— y el gradiente
 * sigue saliendo del mismo codigo verificado.
 */
inline VarPtr Conv2DVar(const VarPtr& entrada, const VarPtr& pesos, int stride = 1,
                        int padding = 0) {
  const std::vector<int>& fp = pesos->Shape();
  if (fp.size() != 4 || fp[2] != fp[3]) {
    throw std::invalid_argument("Conv2DVar: los pesos deben ser [salida, entrada, k, k].");
  }

  // La capa guarda los pesos como parametro propio; aqui se le inyectan los del
  // grafo antes de cada pasada para que ambos vean lo mismo.
  auto capa = std::make_shared<Conv2D>(fp[1], fp[0], fp[2], stride, padding);
  std::memcpy(capa->Weight().Data(), pesos->Value().Data(),
              pesos->Value().TotalSize() * sizeof(float));
  capa->Bias().Zeros();

  Tensor out = capa->Forward(entrada->Value());
  return MakeOp(std::move(out), {entrada, pesos},
                [entrada, pesos, capa](const Tensor& g) {
                  const Tensor dx = capa->Backward(g);
                  if (entrada->RequiresGrad()) entrada->AccumulateGrad(dx);
                  if (pesos->RequiresGrad()) pesos->AccumulateGrad(*capa->GetGradients()[0]);
                });
}

/**
 * @brief LayerNorm compuesta de primitivas, sin backward propio.
 *
 * Esta es la razon de tener un autograd. La version escrita a mano de esta
 * misma operacion necesita una formula de tres terminos para `dx`, y omitir uno
 * de ellos produce un gradiente equivocado que no da ningun sintoma: fue una de
 * las mutaciones con las que validamos su prueba. Aqui se declara el calculo
 * hacia delante y la derivada sale sola.
 */
inline VarPtr LayerNorm(const VarPtr& x, const VarPtr& gamma, const VarPtr& beta,
                        float eps = 1e-5f) {
  auto mean = MeanLastAxis(x);
  auto centered = Sub(x, mean);
  auto variance = MeanLastAxis(Mul(centered, centered));
  auto denom = Sqrt(AddScalar(variance, eps));
  auto normalized = Div(centered, denom);
  return Add(Mul(normalized, gamma), beta);
}

// Operadores para que las expresiones se lean como la formula.
inline VarPtr operator+(const VarPtr& a, const VarPtr& b) { return Add(a, b); }
inline VarPtr operator-(const VarPtr& a, const VarPtr& b) { return Sub(a, b); }
inline VarPtr operator*(const VarPtr& a, const VarPtr& b) { return Mul(a, b); }

}  // namespace autograd
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_AUTOGRAD_H_
