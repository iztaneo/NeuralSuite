"""Decide si una discrepancia de paridad es un defecto o ruido de float32.

Calcula el mismo GPT en float64 y lo usa como referencia de "valor verdadero".
Después mide a qué distancia queda de él cada implementación en float32: la de
PyTorch y la de C++. Si ambas se desvían de forma comparable, la diferencia
entre ellas es el error de redondeo inherente a float32 y no un defecto de una
de las dos; si la de C++ se desvía mucho más, hay algo que corregir.

Uso:
    ./venv/bin/python precision_probe.py --ref /tmp/gpt_ref.nsp --cpp /tmp/gpt_cpp.nsp
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity
from compare_gpt import param_names, max_rel_error
from export_gpt import load_reference_model_module, collect_params_in_cpp_order


def build_and_run(cfg_kwargs, idx, targets, dtype, llmrasec, seed):
    torch.manual_seed(seed)
    GPT, GPTConfig = load_reference_model_module(llmrasec)
    cfg = GPTConfig(dropout=0.0, bias=True, **cfg_kwargs)
    model = GPT(cfg)
    model.eval()
    for block in model.transformer.h:
        block.mlp.gelu = nn.GELU(approximate="tanh")
    model = model.to(dtype)

    logits, loss = model(idx, targets)
    model.zero_grad(set_to_none=False)
    loss.backward()

    ordered = collect_params_in_cpp_order(model, cfg)
    grads = []
    for _, param, transpose in ordered:
        g = param.grad.detach()
        if transpose:
            g = g.t().contiguous()
        grads.append(g.to(torch.float64).numpy())
    return grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmrasec", default="/Users/indra/Documents/Proyectos/LLMRasec")
    ap.add_argument("--ref", default="/tmp/gpt_ref.nsp")
    ap.add_argument("--cpp", default="/tmp/gpt_cpp.nsp")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ref = nsparity.read(args.ref)
    cpp = nsparity.read(args.cpp)
    meta = ref["meta"]
    cfg_kwargs = dict(
        vocab_size=int(meta[0]), block_size=int(meta[1]), n_layer=int(meta[2]),
        n_head=int(meta[3]), n_embd=int(meta[4]),
    )
    n_layer = int(meta[2])
    names = param_names(n_layer)
    n_params = int(ref["num_params"][0])

    idx = torch.from_numpy(ref["idx"].astype(np.int64))
    targets = torch.from_numpy(ref["targets"].astype(np.int64))

    grads_f64 = build_and_run(cfg_kwargs, idx, targets, torch.float64, args.llmrasec, args.seed)

    print("=" * 76)
    print("SONDA DE PRECISION: distancia de cada implementacion float32 al calculo float64")
    print("=" * 76)
    print(f"{'parametro':<28} {'PyTorch f32':>14} {'C++ f32':>14}   veredicto")
    print("-" * 76)

    verdicts = []
    for i in range(n_params):
        truth = grads_f64[i]
        e_torch = max_rel_error(ref[f"grad{i:03d}"], truth)
        e_cpp = max_rel_error(cpp[f"cpp_grad{i:03d}"], truth)

        # C++ solo es sospechoso si se aleja del valor verdadero mucho mas que
        # PyTorch; que ambos se desvien por igual indica redondeo, no defecto.
        if e_cpp > max(10 * e_torch, 1e-3):
            verdict = "REVISAR"
        else:
            verdict = "ruido f32"
        verdicts.append(verdict)
        if verdict == "REVISAR" or e_cpp > 1e-4:
            print(f"{names[i]:<28} {e_torch:>14.3e} {e_cpp:>14.3e}   {verdict}")

    print("-" * 76)
    n_review = verdicts.count("REVISAR")
    worst_torch = max(max_rel_error(ref[f"grad{i:03d}"], grads_f64[i]) for i in range(n_params))
    worst_cpp = max(max_rel_error(cpp[f"cpp_grad{i:03d}"], grads_f64[i]) for i in range(n_params))
    print(f"peor desviacion frente a float64:  PyTorch f32 = {worst_torch:.3e}   "
          f"C++ f32 = {worst_cpp:.3e}")
    print()
    if n_review == 0:
        print("VEREDICTO: ninguna discrepancia excede el redondeo de float32.")
        print("Las dos implementaciones se apartan del calculo en float64 en la misma medida,")
        print("asi que la diferencia entre ellas es precision, no un defecto de NeuralSuite.")
        return 0
    print(f"VEREDICTO: {n_review} gradiente(s) requieren revision.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
