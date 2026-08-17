// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file module.h
 * @brief Base con registro de parametros y submodulos.
 */

#ifndef NEURAL_SUITE_INCLUDE_MODULE_H_
#define NEURAL_SUITE_INCLUDE_MODULE_H_

#include <string>
#include <utility>
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
    std::vector<Parameter*> all;
    for (const auto& entry : own_parameters_) all.push_back(entry.second);
    for (const auto& child : children_) {
      std::vector<Parameter*> sub = child.second->Parameters();
      all.insert(all.end(), sub.begin(), sub.end());
    }
    return all;
  }

  /**
   * @brief Igual que Parameters(), pero con la ruta de cada uno en el arbol.
   *
   * Las rutas quedan del estilo `blocks.0.attn.c_attn.weight`. Un archivo de
   * pesos que las guarde puede comprobar que corresponde al modelo que se esta
   * cargando, en vez de aceptar cualquier secuencia de numeros del tamano
   * adecuado.
   */
  [[nodiscard]] std::vector<std::pair<std::string, Parameter*>> NamedParameters(
      const std::string& prefix = "") {
    std::vector<std::pair<std::string, Parameter*>> all;
    for (const auto& entry : own_parameters_) {
      all.emplace_back(prefix + entry.first, entry.second);
    }
    for (const auto& child : children_) {
      auto sub = child.second->NamedParameters(prefix + child.first + ".");
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
  void Register(Parameter* p, const std::string& name = "") {
    own_parameters_.emplace_back(
        name.empty() ? "param" + std::to_string(own_parameters_.size()) : name, p);
  }

  /** @brief Declara un submodulo, cuyos parametros pasan a formar parte de este. */
  void Register(Module* m, const std::string& name = "") {
    children_.emplace_back(
        name.empty() ? "sub" + std::to_string(children_.size()) : name, m);
  }

 private:
  std::vector<std::pair<std::string, Parameter*>> own_parameters_;
  std::vector<std::pair<std::string, Module*>> children_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_MODULE_H_
