"""Compara la capa LSTM de C++ contra nn.LSTM de PyTorch.

Contrasta la salida de la secuencia completa, la pérdida, el gradiente de
entrada y los cuatro gradientes de parámetros.

Uso:
    ./venv/bin/python compare_lstm.py --ref /tmp/lstm_ref.nsp --cpp /tmp/lstm_cpp.nsp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity
from compare_gpt import max_rel_error

NAMES = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/lstm_ref.nsp")
    ap.add_argument("--cpp", default="/tmp/lstm_cpp.nsp")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ref = nsparity.read(args.ref)
    cpp = nsparity.read(args.cpp)
    meta = ref["meta"]

    print("=" * 68)
    print("PARIDAD LSTM: nn.LSTM (PyTorch) frente a LSTM (NeuralSuite)")
    print("=" * 68)
    print(f"seq={int(meta[0])} batch={int(meta[1])} input={int(meta[2])} hidden={int(meta[3])}")
    print()

    failures = []

    def check(label, a, b):
        err = max_rel_error(a, b)
        ok = err < args.tol
        if not ok:
            failures.append(label)
        print(f"[{'OK ' if ok else 'FALLA'}] {label:<22} error rel max={err:.3e}")
        return err

    loss_ref = float(ref["ref_loss"][0])
    loss_cpp = float(cpp["cpp_loss"][0])
    loss_err = abs(loss_ref - loss_cpp) / max(abs(loss_ref), 1e-8)
    if loss_err >= args.tol:
        failures.append("loss")
    print(f"[{'OK ' if loss_err < args.tol else 'FALLA'}] {'loss':<22} "
          f"PyTorch={loss_ref:.8f}  C++={loss_cpp:.8f}  error rel={loss_err:.3e}")

    check("salida (h_t)", ref["ref_out"], cpp["cpp_out"])
    check("dx (hacia la entrada)", ref["ref_dx"], cpp["cpp_dx"])
    print()
    print("Gradientes de parametros:")
    for i, name in enumerate(NAMES):
        check("  " + name, ref[f"grad{i:03d}"], cpp[f"cpp_grad{i:03d}"])

    print()
    print("=" * 68)
    if failures:
        print(f"RESULTADO: {len(failures)} discrepancia(s) por encima de {args.tol:g}")
        print("  " + ", ".join(f.strip() for f in failures))
        return 1
    print(f"RESULTADO: PARIDAD CONFIRMADA (tolerancia {args.tol:g})")
    print("La capa LSTM de C++ reproduce nn.LSTM en forward y en backward.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
