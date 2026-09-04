// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de losses.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "losses.h"

#include <stdexcept>
#include <string>

namespace neuralsuite {

float CrossEntropyLoss::Forward(const Tensor& predictions, const Tensor& targets) {
    last_preds_ = predictions;
    last_targets_ = targets;

    // Sin estas comprobaciones el framework acepta matematicas sin sentido y
    // devuelve un numero verosimil, que es la peor forma de fallar en aprendizaje
    // automatico. Ocurrio de verdad: `eval_llm` paso los logits en 3D
    // [lote, pasos, vocabulario] en vez de 2D [N, V]; esto leyo Shape()[0] y
    // Shape()[1] como si fueran N y V, devolvio 12.13 —peor que el azar— y no
    // aviso de nada. Con un modelo que escribia espanol correcto.
    if (predictions.Shape().size() != 2) {
      throw std::invalid_argument(
          "CrossEntropyLoss: las predicciones deben ser [N, clases] y tienen " +
          std::to_string(predictions.Shape().size()) +
          " dimensiones. Aplana los ejes de lote y tiempo antes de llamar.");
    }
    int num_samples = predictions.Shape()[0];
    int num_classes = predictions.Shape()[1];

    if (static_cast<int>(targets.TotalSize()) != num_samples) {
      throw std::invalid_argument(
          "CrossEntropyLoss: hay " + std::to_string(num_samples) +
          " predicciones y " + std::to_string(targets.TotalSize()) + " objetivos.");
    }

    float total_loss = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
      int target_cls = static_cast<int>(targets[i]);
      // Un objetivo fuera de rango no falla: lee la fila de al lado y produce
      // una perdida plausible. Es el mismo modo de fallo silencioso.
      if (target_cls < 0 || target_cls >= num_classes) {
        throw std::out_of_range(
            "CrossEntropyLoss: el objetivo " + std::to_string(target_cls) +
            " en la posicion " + std::to_string(i) + " cae fuera de [0, " +
            std::to_string(num_classes) + ").");
      }

      float max_val = predictions[i * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        if (predictions[i * num_classes + c] > max_val) max_val = predictions[i * num_classes + c];
      }

      float sum_exp = 0.0f;
      for (int c = 0; c < num_classes; ++c) {
        sum_exp += std::exp(predictions[i * num_classes + c] - max_val);
      }

      float log_prob = (predictions[i * num_classes + target_cls] - max_val) - std::log(sum_exp);
      total_loss -= log_prob;
    }
    return total_loss / num_samples;
  }

Tensor CrossEntropyLoss::Backward() {
    int num_samples = last_preds_.Shape()[0];
    int num_classes = last_preds_.Shape()[1];

    Tensor dlogits(last_preds_.Shape());

    for (int i = 0; i < num_samples; ++i) {
      int target_cls = static_cast<int>(last_targets_[i]);

      float max_val = last_preds_[i * num_classes];
      for (int c = 1; c < num_classes; ++c) {
        if (last_preds_[i * num_classes + c] > max_val) max_val = last_preds_[i * num_classes + c];
      }

      float sum = 0.0f;
      for (int c = 0; c < num_classes; ++c) sum += std::exp(last_preds_[i * num_classes + c] - max_val);

      for (int c = 0; c < num_classes; ++c) {
        float prob = std::exp(last_preds_[i * num_classes + c] - max_val) / sum;
        float target_val = (c == target_cls) ? 1.0f : 0.0f;
        dlogits[i * num_classes + c] = (prob - target_val) / num_samples;
      }
    }
    return dlogits;
  }

float MSELoss::Forward(const Tensor& predictions, const Tensor& targets) {
    last_preds_ = predictions;
    last_targets_ = targets;
    size_t sz = predictions.TotalSize();

    float loss = 0.0f;
    for (size_t i = 0; i < sz; ++i) {
      float diff = predictions[i] - targets[i];
      loss += diff * diff;
    }
    return loss / sz;
  }

Tensor MSELoss::Backward() {
    size_t sz = last_preds_.TotalSize();
    Tensor dpreds(last_preds_.Shape());
    for (size_t i = 0; i < sz; ++i) {
      dpreds[i] = 2.0f * (last_preds_[i] - last_targets_[i]) / sz;
    }
    return dpreds;
  }

}  // namespace neuralsuite
