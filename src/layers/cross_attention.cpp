// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/cross_attention.h.

#include "layers/cross_attention.h"

#include <cmath>
#include <cstring>

#include "parallel.h"

namespace neuralsuite {

namespace {

void Comprobar3D(const Tensor& t, const char* quien) {
  if (t.Shape().size() != 3) {
    throw std::invalid_argument(std::string(quien) + ": se esperaba [lote, pasos, "
                                "dimension] y tiene " +
                                std::to_string(t.Shape().size()) + " ejes.");
  }
}

}  // namespace

Tensor CrossAttention::Forward(const Tensor& query, const Tensor& contexto) {
  Comprobar3D(query, "CrossAttention (consulta)");
  Comprobar3D(contexto, "CrossAttention (contexto)");

  lote_ = query.Shape()[0];
  tq_ = query.Shape()[1];
  tc_ = contexto.Shape()[1];

  if (contexto.Shape()[0] != lote_) {
    throw std::invalid_argument("CrossAttention: consulta y contexto no comparten lote.");
  }
  if (query.Shape()[2] != n_embd_ || contexto.Shape()[2] != ctx_dim_) {
    throw std::invalid_argument("CrossAttention: dimensiones que no encajan con la capa.");
  }

  q_ = q_proj_.Forward(query);      // [B, Tq, C]
  k_ = k_proj_.Forward(contexto);   // [B, Tc, C]
  v_ = v_proj_.Forward(contexto);   // [B, Tc, C]

  const int hd = n_embd_ / n_head_;
  const float escala = 1.0f / std::sqrt(static_cast<float>(hd));

  pesos_.Resize({lote_ * n_head_, tq_, tc_});
  Tensor ctx({lote_, tq_, n_embd_});

  // Se reparte por (lote, cabeza): cada par escribe su propio bloque de pesos y
  // su propia porcion del contexto, asi que no hay reduccion entre hilos.
  parallel::ParallelFor(lote_ * n_head_, /*min_per_thread=*/1, [&](int desde, int hasta) {
    for (int bh = desde; bh < hasta; ++bh) {
      const int b = bh / n_head_;
      const int h = bh % n_head_;
      const size_t base_pesos = static_cast<size_t>(bh) * tq_ * tc_;

      // Puntuaciones: Q·Kᵀ escalado. NO se aplica mascara causal: cada posicion
      // de la consulta mira todo el contexto, porque en un prompt no hay un
      // "antes".
      Tensor puntos({tq_, tc_});
      for (int i = 0; i < tq_; ++i) {
        const size_t qi = (static_cast<size_t>(b) * tq_ + i) * n_embd_ + h * hd;
        for (int j = 0; j < tc_; ++j) {
          const size_t kj = (static_cast<size_t>(b) * tc_ + j) * n_embd_ + h * hd;
          float acc = 0.0f;
          for (int d = 0; d < hd; ++d) acc += q_[qi + d] * k_[kj + d];
          puntos[static_cast<size_t>(i) * tc_ + j] = acc * escala;
        }
      }

      Tensor att;
      SoftmaxForward(puntos, att);
      std::memcpy(pesos_.Data() + base_pesos, att.Data(),
                  static_cast<size_t>(tq_) * tc_ * sizeof(float));

      for (int i = 0; i < tq_; ++i) {
        const size_t oi = (static_cast<size_t>(b) * tq_ + i) * n_embd_ + h * hd;
        for (int d = 0; d < hd; ++d) ctx[oi + d] = 0.0f;
        for (int j = 0; j < tc_; ++j) {
          const float a = att[static_cast<size_t>(i) * tc_ + j];
          const size_t vj = (static_cast<size_t>(b) * tc_ + j) * n_embd_ + h * hd;
          for (int d = 0; d < hd; ++d) ctx[oi + d] += a * v_[vj + d];
        }
      }
    }
  });

  return o_proj_.Forward(ctx);
}

Tensor CrossAttention::Backward(const Tensor& dout) {
  const int hd = n_embd_ / n_head_;
  const float escala = 1.0f / std::sqrt(static_cast<float>(hd));

  const Tensor dctx = o_proj_.Backward(dout);   // [B, Tq, C]

  Tensor dq(q_.Shape()), dk(k_.Shape()), dv(v_.Shape());
  dq.Zeros();
  dk.Zeros();
  dv.Zeros();

  // Aqui NO se reparte por (lote, cabeza) aunque se podria: `dk` y `dv` los
  // escriben todas las posiciones de la consulta de esa cabeza, y aunque los
  // bloques por cabeza son disjuntos, el bucle interno acumula sobre `j`, que
  // recorre el contexto entero. Se deja secuencial por fila de consulta para que
  // el orden de acumulacion sea fijo y el resultado no dependa de los hilos.
  for (int bh = 0; bh < lote_ * n_head_; ++bh) {
    const int b = bh / n_head_;
    const int h = bh % n_head_;
    const size_t base_pesos = static_cast<size_t>(bh) * tq_ * tc_;

    for (int i = 0; i < tq_; ++i) {
      const size_t oi = (static_cast<size_t>(b) * tq_ + i) * n_embd_ + h * hd;

      // 1. Gradiente respecto a los pesos de atencion, y de paso dv.
      std::vector<float> datt(tc_, 0.0f);
      for (int j = 0; j < tc_; ++j) {
        const size_t vj = (static_cast<size_t>(b) * tc_ + j) * n_embd_ + h * hd;
        const float a = pesos_[base_pesos + static_cast<size_t>(i) * tc_ + j];
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) {
          acc += dctx[oi + d] * v_[vj + d];
          dv[vj + d] += a * dctx[oi + d];
        }
        datt[j] = acc;
      }

      // 2. Backward del softmax: dp_j = a_j * (datt_j - suma(a * datt)).
      //    Restar esa suma es lo que mantiene la fila normalizada; omitirlo da
      //    un gradiente que apunta en una direccion parecida y esta mal.
      double suma = 0.0;
      for (int j = 0; j < tc_; ++j) {
        suma += static_cast<double>(pesos_[base_pesos + static_cast<size_t>(i) * tc_ + j]) *
                datt[j];
      }
      for (int j = 0; j < tc_; ++j) {
        const float a = pesos_[base_pesos + static_cast<size_t>(i) * tc_ + j];
        const float dp = a * (datt[j] - static_cast<float>(suma)) * escala;
        const size_t qi = oi;
        const size_t kj = (static_cast<size_t>(b) * tc_ + j) * n_embd_ + h * hd;
        for (int d = 0; d < hd; ++d) {
          dq[qi + d] += dp * k_[kj + d];
          dk[kj + d] += dp * q_[qi + d];
        }
      }
    }
  }

  const Tensor d_query = q_proj_.Backward(dq);
  const Tensor dc_k = k_proj_.Backward(dk);
  const Tensor dc_v = v_proj_.Backward(dv);

  // El contexto alimenta K y V, asi que recibe la SUMA de las dos ramas.
  // Quedarse con una sola es el error que deja al codificador de texto
  // entrenando a la mitad, sin que nada falle.
  d_contexto_ = dc_k;
  for (size_t i = 0; i < d_contexto_.TotalSize(); ++i) d_contexto_[i] += dc_v[i];

  return d_query;
}

}  // namespace neuralsuite
