"""Exporta un GPT de PyTorch (LLMRasec) para compararlo con el GPT de C++.

Escribe los pesos en el orden exacto que devuelve `GPTModel::GetParameters()`
en C++, junto con la entrada, los logits, la pérdida y los gradientes que
produce PyTorch. El binario de paridad en C++ carga ese archivo, repite el
cálculo y se contrasta con `compare_gpt.py`.

Diferencias que se controlan aquí para que la comparación sea justa:

- Dropout a cero y `model.eval()`: en otro caso el forward de referencia es
  estocástico y ninguna comparación tiene sentido.
- GELU: la implementación de C++ usa la aproximación por tanh, mientras que
  `nn.GELU()` de PyTorch usa por defecto la formulación exacta con erf. Con
  --gelu tanh se sustituye por la aproximación para aislar esa diferencia de
  cualquier otra.
- Disposición de los pesos: `nn.Linear` guarda [out, in] y calcula x·Wᵀ; la
  capa `Linear` de C++ guarda [in, out] y calcula x·W. Todas las matrices de
  capas densas se transponen al exportar.

Uso:
    ./venv/bin/python export_gpt.py --out /tmp/gpt_ref.nsp [--gelu tanh]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity


def load_reference_model_module(llmrasec_dir):
    """Importa `src.model` del proyecto PyTorch de referencia."""
    if not os.path.isdir(llmrasec_dir):
        raise SystemExit(
            f"No se encontro el proyecto de referencia en {llmrasec_dir}.\n"
            "Indicalo con --llmrasec."
        )
    sys.path.insert(0, llmrasec_dir)
    from src.model import GPT, GPTConfig  # noqa: E402

    return GPT, GPTConfig


def collect_params_in_cpp_order(model, cfg):
    """Devuelve los parámetros en el orden de `GPTModel::GetParameters()`.

    C++ recorre: wte, wpe, y por cada bloque ln_1(gamma,beta),
    attn(c_attn.W, c_attn.b, c_proj.W, c_proj.b), ln_2(gamma,beta),
    mlp_fc(W,b), mlp_proj(W,b); y termina con ln_f(gamma,beta).
    """
    t = model.transformer
    ordered = []

    ordered.append(("wte.weight", t.wte.weight, False))
    ordered.append(("wpe.weight", t.wpe.weight, False))

    for i, block in enumerate(t.h):
        p = f"block{i}."
        ordered.append((p + "ln_1.gamma", block.ln_1.weight, False))
        ordered.append((p + "ln_1.beta", block.ln_1.bias, False))
        # Las matrices densas se transponen: nn.Linear guarda [out, in].
        ordered.append((p + "attn.c_attn.weight", block.attn.c_attn.weight, True))
        ordered.append((p + "attn.c_attn.bias", block.attn.c_attn.bias, False))
        ordered.append((p + "attn.c_proj.weight", block.attn.c_proj.weight, True))
        ordered.append((p + "attn.c_proj.bias", block.attn.c_proj.bias, False))
        ordered.append((p + "ln_2.gamma", block.ln_2.weight, False))
        ordered.append((p + "ln_2.beta", block.ln_2.bias, False))
        ordered.append((p + "mlp_fc.weight", block.mlp.c_fc.weight, True))
        ordered.append((p + "mlp_fc.bias", block.mlp.c_fc.bias, False))
        ordered.append((p + "mlp_proj.weight", block.mlp.c_proj.weight, True))
        ordered.append((p + "mlp_proj.bias", block.mlp.c_proj.bias, False))

    ordered.append(("ln_f.gamma", t.ln_f.weight, False))
    ordered.append(("ln_f.beta", t.ln_f.bias, False))
    return ordered


def main():
    # Por defecto se busca el proyecto de referencia junto a este repositorio.
    # Nunca una ruta absoluta: dependeria de la maquina de quien lo escribio.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_llmrasec = os.path.join(os.path.dirname(repo_root), "LLMRasec")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmrasec", default=default_llmrasec)
    ap.add_argument("--out", default="/tmp/gpt_ref.nsp")
    ap.add_argument("--vocab_size", type=int, default=11)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--n_embd", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gelu", choices=["exact", "tanh"], default="tanh",
                    help="tanh iguala la aproximacion que usa C++; exact usa erf")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    GPT, GPTConfig = load_reference_model_module(args.llmrasec)

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,   # el forward debe ser determinista
        bias=True,
    )
    model = GPT(cfg)
    model.eval()      # desactiva dropout aunque la probabilidad ya sea cero

    if args.gelu == "tanh":
        for block in model.transformer.h:
            block.mlp.gelu = nn.GELU(approximate="tanh")

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), generator=g)
    targets = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), generator=g)

    logits, loss = model(idx, targets)
    model.zero_grad(set_to_none=False)
    loss.backward()

    tensors = {
        "meta": np.array(
            [cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd,
             args.batch, args.seq],
            dtype=np.float32,
        ),
        "idx": idx.to(torch.float32).numpy(),
        "targets": targets.to(torch.float32).numpy(),
        "ref_logits": logits.detach().numpy(),
        "ref_loss": np.array([loss.item()], dtype=np.float32),
    }

    ordered = collect_params_in_cpp_order(model, cfg)
    for i, (name, param, transpose) in enumerate(ordered):
        value = param.detach()
        grad = param.grad.detach() if param.grad is not None else torch.zeros_like(value)
        if transpose:
            value = value.t().contiguous()
            grad = grad.t().contiguous()
        tensors[f"param{i:03d}"] = value.numpy()
        tensors[f"grad{i:03d}"] = grad.numpy()

    tensors["num_params"] = np.array([len(ordered)], dtype=np.float32)

    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  config      : vocab={cfg.vocab_size} block={cfg.block_size} "
          f"n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd}")
    print(f"  entrada     : [{args.batch}, {args.seq}]")
    print(f"  GELU        : {args.gelu}")
    print(f"  parametros  : {len(ordered)}")
    print(f"  loss ref    : {loss.item():.8f}")


if __name__ == "__main__":
    main()
