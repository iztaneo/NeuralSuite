// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de gpt.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "gpt.h"

namespace neuralsuite {

Tensor GPTBlock::Forward(const Tensor& input) {
    Tensor x_norm1 = ln_1_.Forward(input);
    Tensor attn_out = attn_.Forward(x_norm1);
    Tensor x1(input.Shape());
    ElementwiseAdd(input, attn_out, x1);

    Tensor x_norm2 = ln_2_.Forward(x1);
    Tensor fc_out = mlp_fc_.Forward(x_norm2);
    Tensor gelu_out = mlp_gelu_.Forward(fc_out);
    Tensor proj_out = mlp_proj_.Forward(gelu_out);
    Tensor x2(x1.Shape());
    ElementwiseAdd(x1, proj_out, x2);

    return x2;
  }

Tensor GPTBlock::Backward(const Tensor& dout) {
    // 1. Backward del bloque MLP con conexión residual
    Tensor dproj_out = mlp_proj_.Backward(dout);
    Tensor dgelu_out = mlp_gelu_.Backward(dproj_out);
    Tensor dfc_out = mlp_fc_.Backward(dgelu_out);
    Tensor dx_norm2 = ln_2_.Backward(dfc_out);

    Tensor dx1(dout.Shape());
    ElementwiseAdd(dout, dx_norm2, dx1);

    // 2. Backward de la Atención Multi-Cabeza con conexión residual
    Tensor dattn_out = attn_.Backward(dx1);
    Tensor dx_norm1 = ln_1_.Backward(dattn_out);

    Tensor dx(dx1.Shape());
    ElementwiseAdd(dx1, dx_norm1, dx);

    return dx;
  }

Tensor GPTBlock::ForwardWithKVCache(const Tensor& input) {
    Tensor x_norm1 = ln_1_.Forward(input);
    Tensor attn_out = attn_.ForwardWithKVCache(x_norm1);
    Tensor x1(input.Shape());
    ElementwiseAdd(input, attn_out, x1);

    Tensor x_norm2 = ln_2_.Forward(x1);
    Tensor fc_out = mlp_fc_.Forward(x_norm2);
    Tensor gelu_out = mlp_gelu_.Forward(fc_out);
    Tensor proj_out = mlp_proj_.Forward(gelu_out);
    Tensor x2(x1.Shape());
    ElementwiseAdd(x1, proj_out, x2);

    return x2;
  }

void GPTModel::ClearKVCache() {
    for (auto& block : blocks_) {
      block->ClearKVCache();
    }
  }

Tensor GPTModel::ForwardWithKVCache(int token_idx, int pos_idx) {
    Tensor tok_tensor({1, 1});
    tok_tensor[0] = static_cast<float>(token_idx);
    Tensor tok_emb = wte_.Forward(tok_tensor);

    Tensor pos_tensor({1, 1});
    pos_tensor[0] = static_cast<float>(pos_idx);
    Tensor pos_emb = wpe_.Forward(pos_tensor);

    Tensor x({1, 1, config_.n_embd});
    for (int d = 0; d < config_.n_embd; ++d) {
      x[d] = tok_emb[d] + pos_emb[d];
    }

    for (auto& block : blocks_) {
      x = block->ForwardWithKVCache(x);
    }

    x = ln_f_.Forward(x);

    Tensor x_2d({1, config_.n_embd});
    std::memcpy(x_2d.Data(), x.Data(), config_.n_embd * sizeof(float));

    Tensor wte_T = Transpose(wte_.Weight());
    Tensor logits_2d;
    MatMul(x_2d, wte_T, logits_2d);

    Tensor logits({1, 1, config_.vocab_size});
    std::memcpy(logits.Data(), logits_2d.Data(), config_.vocab_size * sizeof(float));

    return logits;
  }

Tensor GPTModel::Forward(const Tensor& idx) {

    int batch_size = idx.Shape()[0];
    int seq_len = idx.Shape()[1];

    Tensor tok_emb = wte_.Forward(idx);

    Tensor pos_idx({1, seq_len});
    for (int t = 0; t < seq_len; ++t) pos_idx[t] = static_cast<float>(t);
    Tensor pos_emb = wpe_.Forward(pos_idx);

    Tensor x({batch_size, seq_len, config_.n_embd});
    for (int b = 0; b < batch_size; ++b) {
      for (int t = 0; t < seq_len; ++t) {
        for (int d = 0; d < config_.n_embd; ++d) {
          size_t idx_3d = (b * seq_len + t) * config_.n_embd + d;
          size_t pos_3d = t * config_.n_embd + d;
          x[idx_3d] = tok_emb[idx_3d] + pos_emb[pos_3d];
        }
      }
    }

    for (auto& block : blocks_) {
      x = block->Forward(x);
    }

    x = ln_f_.Forward(x);

    // last_x_2d_ es una instantanea: el backward la necesita despues de que
    // x haya sido reemplazado, asi que aqui si corresponde copiar.
    last_x_2d_ = x;
    last_x_2d_.Reshape({batch_size * seq_len, config_.n_embd});

    // Weight Tying: logits = x_2d * W_wte^T
    Tensor wte_T = Transpose(wte_.Weight());
    Tensor logits_2d;
    MatMul(last_x_2d_, wte_T, logits_2d);

    logits_2d.Reshape({batch_size, seq_len, config_.vocab_size});
    return logits_2d;
  }

Tensor GPTModel::Backward(const Tensor& dlogits) {
    int batch_size = dlogits.Shape()[0];
    int seq_len = dlogits.Shape()[1];

    const Tensor dlogits_2d = dlogits.View({batch_size * seq_len, config_.vocab_size});

    // Weight Tying Backward:
    // dx_2d = dlogits_2d * W_wte
    Tensor dx_2d;
    MatMul(dlogits_2d, wte_.Weight(), dx_2d);

    // Contribución de la cabeza de salida: dW_output = dlogits_2d^T * x_2d.
    // Se calcula aquí (necesita last_x_2d_) pero se acumula al final: la matriz
    // wte_ está compartida entre el embedding de entrada y la cabeza de salida,
    // y Embedding::Backward() reinicia su gradiente a cero. Acumular antes de
    // esa llamada haría que la contribución se perdiera por completo.
    Tensor dlogits_2d_T = Transpose(dlogits_2d);
    Tensor dwte_output;
    MatMul(dlogits_2d_T, last_x_2d_, dwte_output);

    dx_2d.Reshape({batch_size, seq_len, config_.n_embd});
    Tensor dx = ln_f_.Backward(dx_2d);

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
      dx = (*it)->Backward(dx);
    }

    // Escribe la contribución del embedding de entrada en dweight_.
    wte_.Backward(dx);
    wpe_.Backward(dx);

    // dW_total = dW_embedding + dW_output.
    Tensor& wte_grad = wte_.WeightParam().Grad();
    for (size_t i = 0; i < wte_grad.TotalSize(); ++i) {
      wte_grad[i] += dwte_output[i];
    }

    return dx;
  }

std::vector<Tensor*> GPTModel::GetParameters() {
    std::vector<Tensor*> out;
    for (Parameter* p : Parameters()) out.push_back(&p->Value());
    return out;
  }

std::vector<Tensor*> GPTModel::GetGradients() {
    std::vector<Tensor*> out;
    for (Parameter* p : Parameters()) out.push_back(&p->Grad());
    return out;
  }

bool GPTModel::SaveWeights(const std::string& filepath) {
    const auto result = nsf::Save(filepath, nsf::FromNamedParameters(NamedParameters()),
                                 ArchitectureMetadata());
    if (!result) std::cerr << "Error al guardar: " << result.error << "\n";
    return result.ok;
  }

bool GPTModel::LoadWeights(const std::string& filepath) {
    const auto result = nsf::Load(filepath, nsf::FromNamedParameters(NamedParameters()),
                                 ArchitectureMetadata());
    if (!result) std::cerr << "Error al cargar: " << result.error << "\n";
    return result.ok;
  }

std::map<std::string, std::string> GPTModel::ArchitectureMetadata() const {
    return {{"arch", "gpt"},
            {"vocab_size", std::to_string(config_.vocab_size)},
            {"block_size", std::to_string(config_.block_size)},
            {"n_layer", std::to_string(config_.n_layer)},
            {"n_head", std::to_string(config_.n_head)},
            {"n_embd", std::to_string(config_.n_embd)}};
  }

}  // namespace neuralsuite
