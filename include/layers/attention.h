// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file attention.h
 * @brief Causal Multi-Head Self-Attention with Full Linear Projection and Exact Backward Pass.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parallel.h"
#include "linear.h"

namespace neuralsuite {

/**
 * @class MultiHeadAttentionReference
 * @brief La atencion escrita como su definicion, elemento a elemento.
 *
 * Se conserva como oraculo, igual que Conv2DReference y LSTMReference. Es lenta
 * -1.3 GFLOP/s en un solo hilo- pero se lee al lado de las formulas y no tiene
 * ninguna reordenacion de memoria que pueda desalinearse.
 *
 * @class MultiHeadAttention
 * @brief Multi-Head Causal Self-Attention Layer for Transformer Decoders.
 *
 * La posicion llega por embeddings aprendidos (`wpe_` del modelo), no por
 * rotacion de Q y K. Existio aqui un `ApplyRoPE()` que ningun forward llamaba:
 * sugeria una capacidad que el modelo no tiene, asi que se retiro. RoPE sigue
 * planteado en docs/FUTURE_PLAN_KVCACHE_ROPE.md, y anadirlo exige rotar Q y K
 * en el forward, propagar por esa rotacion en el backward con su gradient
 * check, y actualizar la implementacion de referencia en PyTorch para que la
 * comparacion siga siendo valida.
 */
class MultiHeadAttentionReference : public Layer {
 public:
  MultiHeadAttentionReference(int embd, int heads)
      : n_embd_(embd),
        n_head_(heads),
        head_dim_(embd / heads),
        // 0.02 es la convencion de GPT-2 para todas sus densas. Al pasar Linear
        // a Xavier por defecto se actualizo gpt.h y se olvido este archivo, de
        // modo que la atencion quedo con una inicializacion distinta del resto
        // del modelo sin que nada lo senalara.
        c_attn_(embd, 3 * embd, /*init_std=*/0.02f),
        c_proj_(embd, embd, /*init_std=*/0.02f) {
    Register(&c_attn_, "c_attn");
    Register(&c_proj_, "c_proj");
    // Si embd no es múltiplo de heads, head_dim_ trunca y las dimensiones
    // sobrantes quedan sin asignar a ninguna cabeza, en silencio.
    if (embd <= 0 || heads <= 0) {
      throw std::invalid_argument("MultiHeadAttentionReference: embd y heads deben ser positivos.");
    }
    if (embd % heads != 0) {
      throw std::invalid_argument(
          "MultiHeadAttention: n_embd (" + std::to_string(embd) +
          ") debe ser multiplo de n_head (" + std::to_string(heads) + ").");
    }
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  void ClearKVCache();

  Tensor ForwardWithKVCache(const Tensor& single_token_input);

 private:
  int n_embd_;
  int n_head_;
  int head_dim_;

  Linear c_attn_;
  Linear c_proj_;

  Tensor last_input_;
  Tensor qkv_cache_;
  Tensor attn_probs_cache_;

  std::vector<std::vector<float>> k_cache_;
  std::vector<std::vector<float>> v_cache_;
};


/**
 * @class MultiHeadAttention
 * @brief La misma atencion, resuelta con multiplicaciones de matrices.
 *
 * Dentro de la atencion hay dos productos de matrices por cada cabeza:
 *
 *     puntuaciones = Q · K^T        salida = P · V
 *
 * y en `MultiHeadAttentionReference` estan escritos como bucles que recorren
 * cada elemento. Eso daba 1.3 GFLOP/s teniendo un `MatMul` que alcanza 232 y
 * diez nucleos sin usar. Es el tercer y ultimo caso del mismo patron, despues
 * de la convolucion y la celda recurrente.
 *
 * El coste crece con el cuadrado de la longitud del contexto, y eso **no se
 * arregla**: cada posicion mira a todas las anteriores, esa es la definicion
 * del mecanismo. Lo que se arregla es la constante. Medido sobre el GPT de
 * `train_llm`, pasar de contexto 64 a 256 multiplicaba el tiempo por trece
 * cuando lo inevitable era cuatro.
 *
 * La reordenacion que hace falta: `c_attn_` entrega Q, K y V entrelazados en
 * una sola fila de `3 * n_embd`, y cada cabeza ocupa un tramo de esa fila. Para
 * multiplicar hay que extraer cada cabeza como una matriz contigua `[T,
 * head_dim]`. Es el mismo trabajo que hace `im2col` en la convolucion: copiar
 * para poder leer seguido.
 *
 * Se calcula el cuadrado completo de puntuaciones, incluida la mitad que la
 * mascara causal descarta. Es el doble de operaciones que el bucle triangular
 * del oraculo, y aun asi sale mucho mas rapido: un `MatMul` sobre memoria
 * contigua rinde mas que la mitad de las operaciones hechas a saltos.
 * `CausalSoftmaxForward` pone a cero el triangulo superior, de modo que la
 * mascara se aplica sola y el gradiente hereda esos ceros sin tener que
 * enmascararlo aparte.
 *
 * El reparto entre hilos va por pareja (muestra, cabeza). Son independientes y
 * -esto es lo que permite hacerlo sin reduccion- cada una escribe en un tramo
 * distinto de `dqkv`: la cabeza `h` solo toca las columnas `h*head_dim` en
 * adelante. No hay dos hilos que sumen sobre la misma posicion, asi que el
 * resultado es identico con uno o con diez.
 */
class MultiHeadAttention : public Layer {
 public:
  MultiHeadAttention(int n_embd, int n_head)
      : n_embd_(n_embd),
        n_head_(n_head),
        head_dim_(n_embd / n_head),
        c_attn_(n_embd, 3 * n_embd, /*init_std=*/0.02f),
        c_proj_(n_embd, n_embd, /*init_std=*/0.02f) {
    if (n_head <= 0 || n_embd <= 0) {
      throw std::invalid_argument("MultiHeadAttention: n_embd y n_head deben ser positivos.");
    }
    if (n_embd % n_head != 0) {
      throw std::invalid_argument(
          "MultiHeadAttention: n_embd (" + std::to_string(n_embd) +
          ") debe ser divisible entre n_head (" + std::to_string(n_head) + ").");
    }
    Register(&c_attn_, "c_attn");
    Register(&c_proj_, "c_proj");
  }

  Tensor Forward(const Tensor& input) override;

  Tensor Backward(const Tensor& dout) override;

  [[nodiscard]] const Linear& AttnProjection() const { return c_attn_; }
  Linear& AttnProjection() { return c_attn_; }
  [[nodiscard]] const Linear& OutProjection() const { return c_proj_; }
  Linear& OutProjection() { return c_proj_; }

  /**
   * @brief Un solo token, reutilizando las claves y valores ya calculados.
   *
   * Este camino se queda escalar a proposito: procesa **un** token, de modo que
   * no hay matrices que multiplicar y reordenar la memoria costaria mas que el
   * propio calculo. Es el mismo codigo que la version de referencia.
   */
  Tensor ForwardWithKVCache(const Tensor& single_token_input);

  void ClearKVCache();


 private:
  /**
   * @brief Saca Q, K^T y V de una cabeza como matrices contiguas.
   *
   * `c_attn_` los entrega entrelazados —cada fila lleva los tres, y dentro de
   * cada uno un tramo por cabeza—, que es una disposicion imposible de
   * multiplicar sin saltar por la memoria. K sale ya transpuesta porque asi es
   * como entra en el producto.
   */
  void Extraer(int b, int h, int T, Tensor* Q, Tensor* Kt, Tensor* V) const;

  int n_embd_;
  int n_head_;
  int head_dim_;
  Linear c_attn_;
  Linear c_proj_;

  Tensor last_input_;
  Tensor qkv_cache_;
  Tensor attn_probs_cache_;
  std::vector<std::vector<float>> k_cache_;
  std::vector<std::vector<float>> v_cache_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_ATTENTION_H_
