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
 *     auto x = Variable::Create(tensor, /*requires_grad=*\/ true);
 *     auto y = Tanh(MatMul(x, w) + b);
 *     Backward(Sum(y));
 *     x->Grad();   // ya calculado, sin escribir ninguna derivada
 *
 * Limitaciones deliberadas de esta primera version, para no acumular
 * complejidad antes de tener las primitivas verificadas:
 *
 * - Sin broadcasting: las operaciones elemento a elemento exigen formas
 *   identicas. Anadirlo sin haber comprobado antes lo basico complicaria el
 *   backward de cada operacion.
 * - Solo `float32`.
 * - El grafo se construye siempre; no hay todavia un modo de inferencia que lo
 *   omita.
 */

#ifndef NEURAL_SUITE_INCLUDE_AUTOGRAD_H_
#define NEURAL_SUITE_INCLUDE_AUTOGRAD_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
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

inline void RequireSameShape(const VarPtr& a, const VarPtr& b, const char* op) {
  if (a->Shape() != b->Shape()) {
    throw std::invalid_argument(std::string(op) +
                                ": las formas deben coincidir; no hay broadcasting.");
  }
}

}  // namespace detail

// ============================================================================
// PRIMITIVAS
// ============================================================================

/** @brief Suma elemento a elemento. d(a+b)/da = 1, d(a+b)/db = 1. */
inline VarPtr Add(const VarPtr& a, const VarPtr& b) {
  detail::RequireSameShape(a, b, "Add");
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = a->Value()[i] + b->Value()[i];

  return MakeOp(std::move(out), {a, b}, [a, b](const Tensor& g) {
    if (a->RequiresGrad()) a->AccumulateGrad(g);
    if (b->RequiresGrad()) b->AccumulateGrad(g);
  });
}

/** @brief Producto elemento a elemento. d(a*b)/da = b, d(a*b)/db = a. */
inline VarPtr Mul(const VarPtr& a, const VarPtr& b) {
  detail::RequireSameShape(a, b, "Mul");
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = a->Value()[i] * b->Value()[i];

  return MakeOp(std::move(out), {a, b}, [a, b](const Tensor& g) {
    if (a->RequiresGrad()) {
      Tensor da(a->Shape());
      for (size_t i = 0; i < da.TotalSize(); ++i) da[i] = g[i] * b->Value()[i];
      a->AccumulateGrad(da);
    }
    if (b->RequiresGrad()) {
      Tensor db(b->Shape());
      for (size_t i = 0; i < db.TotalSize(); ++i) db[i] = g[i] * a->Value()[i];
      b->AccumulateGrad(db);
    }
  });
}

/** @brief Resta elemento a elemento. */
inline VarPtr Sub(const VarPtr& a, const VarPtr& b) {
  detail::RequireSameShape(a, b, "Sub");
  Tensor out(a->Shape());
  for (size_t i = 0; i < out.TotalSize(); ++i) out[i] = a->Value()[i] - b->Value()[i];

  return MakeOp(std::move(out), {a, b}, [a, b](const Tensor& g) {
    if (a->RequiresGrad()) a->AccumulateGrad(g);
    if (b->RequiresGrad()) {
      Tensor db(b->Shape());
      for (size_t i = 0; i < db.TotalSize(); ++i) db[i] = -g[i];
      b->AccumulateGrad(db);
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

// Operadores para que las expresiones se lean como la formula.
inline VarPtr operator+(const VarPtr& a, const VarPtr& b) { return Add(a, b); }
inline VarPtr operator-(const VarPtr& a, const VarPtr& b) { return Sub(a, b); }
inline VarPtr operator*(const VarPtr& a, const VarPtr& b) { return Mul(a, b); }

}  // namespace autograd
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_AUTOGRAD_H_
