"""Compara la capa BiLSTM de C++ contra nn.LSTM(bidirectional=True).

Contrasta la salida completa, la pérdida, el gradiente de entrada —que es la
suma de las dos ramas— y los ocho gradientes de parámetros.

Uso:
    ./venv/bin/python compare_bilstm.py --ref /tmp/bilstm_ref.nsp --cpp /tmp/bilstm_cpp.nsp
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity
from compare_gpt import max_rel_error

NAMES = ["forward.weight_ih", "forward.weight_hh", "forward.bias_ih", "forward.bias_hh",
         "reverse.weight_ih", "reverse.weight_hh", "reverse.bias_ih", "reverse.bias_hh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/bilstm_ref.nsp")
    ap.add_argument("--cpp", default="/tmp/bilstm_cpp.nsp")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ref = nsparity.read(args.ref)
    cpp = nsparity.read(args.cpp)
    meta = ref["meta"]
    hidden = int(meta[3])

    print("=" * 72)
    print("PARIDAD BiLSTM: nn.LSTM(bidirectional=True) frente a BiLSTM (NeuralSuite)")
    print("=" * 72)
    print(f"seq={int(meta[0])} batch={int(meta[1])} input={int(meta[2])} hidden={hidden}")
    print()

    failures = []

    def check(label, a, b):
        err = max_rel_error(a, b)
        ok = err < args.tol
        if not ok:
            failures.append(label)
        print(f"[{'OK ' if ok else 'FALLA'}] {label:<26} error rel max={err:.3e}")
        return err

    loss_ref = float(ref["ref_loss"][0])
    loss_cpp = float(cpp["cpp_loss"][0])
    loss_err = abs(loss_ref - loss_cpp) / max(abs(loss_ref), 1e-8)
    if loss_err >= args.tol:
        failures.append("loss")
    print(f"[{'OK ' if loss_err < args.tol else 'FALLA'}] {'loss':<26} "
          f"PyTorch={loss_ref:.8f}  C++={loss_cpp:.8f}  error rel={loss_err:.3e}")

    check("salida completa", ref["ref_out"], cpp["cpp_out"])

    # Las dos mitades por separado: si se hubieran intercambiado al concatenar,
    # el error total podria quedar disimulado, pero cada mitad lo delata.
    out_ref, out_cpp = ref["ref_out"], cpp["cpp_out"]
    check("  mitad directa", out_ref[..., :hidden], out_cpp[..., :hidden])
    check("  mitad inversa", out_ref[..., hidden:], out_cpp[..., hidden:])

    check("dx (hacia la entrada)", ref["ref_dx"], cpp["cpp_dx"])
    print()
    print("Gradientes de parametros:")
    for i, name in enumerate(NAMES):
        check("  " + name, ref[f"grad{i:03d}"], cpp[f"cpp_grad{i:03d}"])

    print()
    print("=" * 72)
    if failures:
        print(f"RESULTADO: {len(failures)} discrepancia(s) por encima de {args.tol:g}")
        print("  " + ", ".join(f.strip() for f in failures))
        return 1
    print(f"RESULTADO: PARIDAD CONFIRMADA (tolerancia {args.tol:g})")
    print("BiLSTM reproduce nn.LSTM bidireccional en forward y en backward.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
