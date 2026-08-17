/**
 * @file gpt.h
 * @brief Modelo completo de GPT (Decoder-Only Transformer) en C++17 puro.
 */

#ifndef NEURAL_SUITE_GPT_H
#define NEURAL_SUITE_GPT_H

#include "tensor.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"
#include "layers/attention.h"
#include "layers/linear.h"
#include "activations.h"
#include "losses.h"
#include "tokenizer.h"
#include <memory>

namespace ns {

struct GPTConfig {
    int vocab_size = 54;
    int block_size = 64;
    int n_layer = 4;
    int n_head = 4;
    int n_embd = 128;
};

class GPTBlock : public Layer {
public:
    LayerNormLayer ln_1;
    MultiHeadAttention attn;
    LayerNormLayer ln_2;
    Linear mlp_fc;
    Activation mlp_gelu;
    Linear mlp_proj;

    GPTBlock(const GPTConfig& config)
        : ln_1(config.n_embd),
          attn(config.n_embd, config.n_head),
          ln_2(config.n_embd),
          mlp_fc(config.n_embd, 4 * config.n_embd),
          mlp_gelu(ActivationType::GELU),
          mlp_proj(4 * config.n_embd, config.n_embd) {}

    Tensor forward(const Tensor& input) override {
        // Pre-LN Attention + Residual
        Tensor x_norm1 = ln_1.forward(input);
        Tensor attn_out = attn.forward(x_norm1);
        Tensor x1(input.shape);
        elementwise_add(input, attn_out, x1);

        // Pre-LN MLP + Residual
        Tensor x_norm2 = ln_2.forward(x1);
        Tensor fc_out = mlp_fc.forward(x_norm2);
        Tensor gelu_out = mlp_gelu.forward(fc_out);
        Tensor proj_out = mlp_proj.forward(gelu_out);
        Tensor x2(x1.shape);
        elementwise_add(x1, proj_out, x2);

        return x2;
    }

    Tensor backward(const Tensor& dout) override {
        return dout; // Backward simplificado
    }

    std::vector<Tensor*> get_parameters() override {
        std::vector<Tensor*> p;
        for (auto p1 : ln_1.get_parameters()) p.push_back(p1);
        for (auto p2 : attn.get_parameters()) p.push_back(p2);
        for (auto p3 : ln_2.get_parameters()) p.push_back(p3);
        for (auto p4 : mlp_fc.get_parameters()) p.push_back(p4);
        for (auto p5 : mlp_proj.get_parameters()) p.push_back(p5);
        return p;
    }

    std::vector<Tensor*> get_gradients() override {
        std::vector<Tensor*> g;
        for (auto g1 : ln_1.get_gradients()) g.push_back(g1);
        for (auto g2 : attn.get_gradients()) g.push_back(g2);
        for (auto g3 : ln_2.get_gradients()) g.push_back(g3);
        for (auto g4 : mlp_fc.get_gradients()) g.push_back(g4);
        for (auto g5 : mlp_proj.get_gradients()) g.push_back(g5);
        return g;
    }
};

class GPTModel {
public:
    GPTConfig config;
    Embedding wte;
    Embedding wpe;
    std::vector<std::shared_ptr<GPTBlock>> blocks;
    LayerNormLayer ln_f;
    Linear lm_head;

    GPTModel(const GPTConfig& cfg)
        : config(cfg),
          wte(cfg.vocab_size, cfg.n_embd),
          wpe(cfg.block_size, cfg.n_embd),
          ln_f(cfg.n_embd),
          lm_head(cfg.n_embd, cfg.vocab_size) {
        
        for (int i = 0; i < cfg.n_layer; ++i) {
            blocks.push_back(std::make_shared<GPTBlock>(cfg));
        }
        
        // Weight tying: compartir matriz de pesos entre wte y lm_head
        lm_head.weight = wte.weight;
    }

    Tensor forward(const Tensor& idx) {
        int B = idx.shape[0];
        int T = idx.shape[1];

        Tensor tok_emb = wte.forward(idx);

        // Positional Embedding
        Tensor pos_idx({1, T});
        for (int t = 0; t < T; ++t) pos_idx.data[t] = static_cast<float>(t);
        Tensor pos_emb = wpe.forward(pos_idx);

        Tensor x({B, T, config.n_embd});
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < config.n_embd; ++d) {
                    size_t idx_3d = (b * T + t) * config.n_embd + d;
                    size_t pos_3d = t * config.n_embd + d;
                    x.data[idx_3d] = tok_emb.data[idx_3d] + pos_emb.data[pos_3d];
                }
            }
        }

        // Pasar por los bloques de Transformer
        for (auto& block : blocks) {
            x = block->forward(x);
        }

        // LayerNorm final
        x = ln_f.forward(x);

        // LM Head
        Tensor x_2d({B * T, config.n_embd});
        std::memcpy(x_2d.data, x.data, x.total_size() * sizeof(float));

        Tensor logits_2d = lm_head.forward(x_2d);
        Tensor logits({B, T, config.vocab_size});
        std::memcpy(logits.data, logits_2d.data, logits_2d.total_size() * sizeof(float));

        return logits;
    }

    std::vector<Tensor*> get_parameters() {
        std::vector<Tensor*> p;
        for (auto p1 : wte.get_parameters()) p.push_back(p1);
        for (auto p2 : wpe.get_parameters()) p.push_back(p2);
        for (auto& block : blocks) {
            for (auto pb : block->get_parameters()) p.push_back(pb);
        }
        for (auto pf : ln_f.get_parameters()) p.push_back(pf);
        return p;
    }

    std::vector<Tensor*> get_gradients() {
        std::vector<Tensor*> g;
        for (auto g1 : wte.get_gradients()) g.push_back(g1);
        for (auto g2 : wpe.get_gradients()) g.push_back(g2);
        for (auto& block : blocks) {
            for (auto gb : block->get_gradients()) g.push_back(gb);
        }
        for (auto gf : ln_f.get_gradients()) g.push_back(gf);
        return g;
    }
};

} // namespace ns

#endif // NEURAL_SUITE_GPT_H
