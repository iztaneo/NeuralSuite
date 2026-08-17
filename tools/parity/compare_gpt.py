"""Compara los resultados de PyTorch y de C++ para el mismo GPT.

Lee el archivo de referencia producido por export_gpt.py y el que escribe el
binario parity_gpt, y contrasta la pérdida, los logits y el gradiente de cada
parámetro. El criterio es el error relativo máximo por tensor, normalizado por
la magnitud de los valores para que los cercanos a cero no dominen.

Uso:
    ./venv/bin/python compare_gpt.py --ref /tmp/gpt_ref.nsp --cpp /tmp/gpt_cpp.nsp
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity

# Los nombres siguen el orden de GPTModel::GetParameters() en C++.
def param_names(n_layer):
    names = ["wte.weight", "wpe.weight"]
    for i in range(n_layer):
        p = f"block{i}."
        names += [
            p + "ln_1.gamma", p + "ln_1.beta",
            p + "attn.c_attn.weight", p + "attn.c_attn.bias",
            p + "attn.c_proj.weight", p + "attn.c_proj.bias",
            p + "ln_2.gamma", p + "ln_2.beta",
            p + "mlp_fc.weight", p + "mlp_fc.bias",
            p + "mlp_proj.weight", p + "mlp_proj.bias",
        ]
    names += ["ln_f.gamma", "ln_f.beta"]
    return names


def max_rel_error(a, b):
    """Error relativo máximo entre dos arrays, normalizado por su magnitud."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-8)
    return float(np.max(np.abs(a - b) / denom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/gpt_ref.nsp")
    ap.add_argument("--cpp", default="/tmp/gpt_cpp.nsp")
    # La tolerancia no es arbitraria: precision_probe.py mide que, para esta
    # configuracion, tanto PyTorch como C++ se apartan del mismo calculo hecho
    # en float64 hasta ~1.8e-4 solo por redondeo. Un umbral mas estricto marca
    # ruido de precision como defecto. Los defectos reales de gradiente que
    # hemos visto aparecen entre 1e-1 y 1.0, muy por encima de este limite.
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="error relativo maximo aceptable (por defecto 1e-3)")
    args = ap.parse_args()

    ref = nsparity.read(args.ref)
    cpp = nsparity.read(args.cpp)

    meta = ref["meta"]
    n_layer = int(meta[2])
    names = param_names(n_layer)
    n_params = int(ref["num_params"][0])

    print("=" * 68)
    print("PARIDAD GPT: PyTorch (LLMRasec) frente a C++ (NeuralSuite)")
    print("=" * 68)
    print(f"config: vocab={int(meta[0])} block={int(meta[1])} n_layer={n_layer} "
          f"n_head={int(meta[3])} n_embd={int(meta[4])}  entrada=[{int(meta[5])}, {int(meta[6])}]")
    print()

    failures = []

    loss_ref = float(ref["ref_loss"][0])
    loss_cpp = float(cpp["cpp_loss"][0])
    loss_err = abs(loss_ref - loss_cpp) / max(abs(loss_ref), 1e-8)
    status = "OK " if loss_err < args.tol else "FALLA"
    if loss_err >= args.tol:
        failures.append("loss")
    print(f"[{status}] loss           PyTorch={loss_ref:.8f}  C++={loss_cpp:.8f}  "
          f"error rel={loss_err:.3e}")

    logits_err = max_rel_error(ref["ref_logits"], cpp["cpp_logits"])
    status = "OK " if logits_err < args.tol else "FALLA"
    if logits_err >= args.tol:
        failures.append("logits")
    print(f"[{status}] logits         error rel max={logits_err:.3e}")
    print()
    print("Gradientes por parametro:")

    worst_name, worst_err = None, 0.0
    for i in range(n_params):
        g_ref = ref[f"grad{i:03d}"]
        g_cpp = cpp[f"cpp_grad{i:03d}"]
        if g_ref.size != g_cpp.size:
            print(f"  [FALLA] {names[i]:<28} tamanos distintos "
                  f"{g_ref.shape} vs {g_cpp.shape}")
            failures.append(names[i])
            continue
        err = max_rel_error(g_ref, g_cpp)
        if err > worst_err:
            worst_err, worst_name = err, names[i]
        if err >= args.tol:
            failures.append(names[i])
            print(f"  [FALLA] {names[i]:<28} error rel max={err:.3e}")

    print(f"  peor gradiente: {worst_name} con error rel {worst_err:.3e}")
    if not [f for f in failures if f not in ("loss", "logits")]:
        print(f"  los {n_params} gradientes por debajo de la tolerancia {args.tol:g}")

    print()
    print("=" * 68)
    if failures:
        print(f"RESULTADO: {len(failures)} discrepancia(s) por encima de {args.tol:g}")
        print("  " + ", ".join(failures[:10]))
        return 1
    print(f"RESULTADO: PARIDAD CONFIRMADA (tolerancia {args.tol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
