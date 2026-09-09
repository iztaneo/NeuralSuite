// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file cross_attention.h
 * @brief Atencion cruzada: la consulta viene de un sitio y las claves de otro.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_CROSS_ATTENTION_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_CROSS_ATTENTION_H_

#include <stdexcept>
#include <string>
#include <vector>

#include "../layer.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class CrossAttention
 * @brief `Q` de una entrada, `K` y `V` de otra.
 *
 * Es la pieza que conecta dos modalidades. En la self-attention que ya existe,
 * `Q`, `K` y `V` salen del mismo tensor; aqui la consulta sale del latente de
 * imagen y las claves y valores del condicionamiento —el texto—, de modo que
 * cada posicion de la imagen puede mirar a todo el prompt.
 *
 * Tres diferencias con `MultiHeadAttention`, y las tres importan:
 *
 * 1. **Dos entradas y dos gradientes.** `Backward` devuelve el de la consulta,
 *    que es lo que exige la interfaz de `Layer`; el del contexto se recoge con
 *    `GradContexto()`. Devolver solo uno y perder el otro cortaria el grafo por
 *    la rama del texto, y el codificador de texto dejaria de entrenar sin que
 *    nada fallara.
 *
 * 2. **No es causal.** Una posicion de la imagen mira todo el contexto, no solo
 *    lo anterior: no hay un "antes" en un prompt. Aplicar la mascara causal aqui
 *    dejaria a los primeros pixeles viendo solo el principio del texto.
 *
 * 3. **Las proyecciones estan separadas.** `MultiHeadAttention` usa una sola
 *    densa que saca `Q`, `K` y `V` de un tirón porque los tres vienen del mismo
 *    tensor. Aqui `Q` sale del latente y `K`,`V` del contexto, que puede tener
 *    otra dimension, asi que van por su cuenta.
 *
 * Las longitudes de consulta y contexto son independientes: `[B, Tq, C]` con
 * `[B, Tc, Cctx]` da `[B, Tq, C]`.
 */
class CrossAttention : public Layer {
 public:
  /** @param ctx_dim dimension del condicionamiento; por defecto, la misma. */
  CrossAttention(int n_embd, int n_head, int ctx_dim = 0)
      : n_embd_(n_embd),
        n_head_(n_head),
        ctx_dim_(ctx_dim > 0 ? ctx_dim : n_embd),
        q_proj_(n_embd, n_embd),
        k_proj_(ctx_dim > 0 ? ctx_dim : n_embd, n_embd),
        v_proj_(ctx_dim > 0 ? ctx_dim : n_embd, n_embd),
        o_proj_(n_embd, n_embd) {
    if (n_head <= 0 || n_embd % n_head != 0) {
      throw std::invalid_argument(
          "CrossAttention: " + std::to_string(n_embd) + " no se reparte en " +
          std::to_string(n_head) + " cabezas iguales.");
    }
    Register(&q_proj_, "q_proj");
    Register(&k_proj_, "k_proj");
    Register(&v_proj_, "v_proj");
    Register(&o_proj_, "o_proj");
  }

  /** @brief `[B, Tq, C]` x `[B, Tc, Cctx]` -> `[B, Tq, C]`. */
  Tensor Forward(const Tensor& query, const Tensor& contexto);

  /**
   * @brief Atajo de self-attention **no causal**: `Forward(x, x)`.
   *
   * Existe para cumplir la interfaz de `Layer`. Que sea no causal es
   * deliberado: es la atencion del centro de una U-Net, donde cada posicion ve
   * toda la imagen. Para la causal esta `MultiHeadAttention`.
   */
  Tensor Forward(const Tensor& input) override { return Forward(input, input); }

  /** @brief Devuelve el gradiente de la CONSULTA; el del contexto va aparte. */
  Tensor Backward(const Tensor& dout) override;

  /** @brief Gradiente del contexto tras `Backward`. */
  [[nodiscard]] const Tensor& GradContexto() const { return d_contexto_; }

  Linear& QProj() { return q_proj_; }
  Linear& KProj() { return k_proj_; }
  Linear& VProj() { return v_proj_; }
  Linear& OProj() { return o_proj_; }

 private:
  int n_embd_;
  int n_head_;
  int ctx_dim_;

  Linear q_proj_, k_proj_, v_proj_, o_proj_;

  // Lo que el backward necesita del forward.
  Tensor q_, k_, v_;        // proyecciones, [B, T*, C]
  Tensor pesos_;            // softmax por (lote, cabeza): [B*H, Tq, Tc]
  Tensor d_contexto_;
  int lote_ = 0, tq_ = 0, tc_ = 0;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_CROSS_ATTENTION_H_
