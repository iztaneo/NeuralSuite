// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file module.h
 * @brief Base con registro de parametros y submodulos.
 */

#ifndef NEURAL_SUITE_INCLUDE_MODULE_H_
#define NEURAL_SUITE_INCLUDE_MODULE_H_

#include <vector>
#include "parameter.h"

namespace neuralsuite {

/**
 * @class Module
 * @brief Contenedor de parametros que recorre sus submodulos automaticamente.
 *
 * Cada modulo declara en su constructor los parametros propios y los submodulos
 * que contiene. `Parameters()` recorre ese arbol, de modo que un modelo compuesto
 * no necesita reimplementar la cascada: antes, cada bloque del GPT enumeraba a
 * mano sus cinco componentes, y una omision ahi era invisible.
 *
 * Los modulos guardan punteros a miembros propios, asi que copiarlos o moverlos
 * dejaria esos punteros apuntando al objeto anterior. Copia y movimiento se
 * eliminan para que el compilador rechace ese uso en lugar de fallar en
 * ejecucion.
 */
class Module {
 public:
  Module() = default;
  virtual ~Module() = default;

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;

  /** @brief Parametros propios y los de todos los submodulos, en orden. */
  [[nodiscard]] std::vector<Parameter*> Parameters() {
    std::vector<Parameter*> all = own_parameters_;
    for (Module* child : children_) {
      std::vector<Parameter*> sub = child->Parameters();
      all.insert(all.end(), sub.begin(), sub.end());
    }
    return all;
  }

  /** @brief Pone a cero el gradiente de todo el arbol. */
  void ZeroGrad() {
    for (Parameter* p : Parameters()) p->ZeroGrad();
  }

 protected:
  /** @brief Declara un parametro propio. El orden de registro es el de salida. */
  void Register(Parameter* p) { own_parameters_.push_back(p); }

  /** @brief Declara un submodulo, cuyos parametros pasan a formar parte de este. */
  void Register(Module* m) { children_.push_back(m); }

 private:
  std::vector<Parameter*> own_parameters_;
  std::vector<Module*> children_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_MODULE_H_
