"""Contrasta RMSNorm y SiLU de C++ contra los de PyTorch."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ref = nsparity.read(args.ref)
    cpp = nsparity.read(args.cpp)

    print("=" * 72)
    print("PARIDAD BLOQUES: RMSNorm, SiLU, GroupNorm y remuestreo 2D frente a NeuralSuite")
    print("=" * 72)
    print(f"  {'tensor':<14} {'error abs':>12} {'error rel':>12}   veredicto")

    peor, peor_nombre = 0.0, ""
    for nombre in ("rms_y", "rms_dx", "rms_dgamma", "silu_y", "silu_dx",
               "gn_y", "gn_dx", "gn_dgamma", "gn_dbeta",
               "up_y", "up_dx", "dn_y", "dn_dx"):
        if nombre not in cpp:
            print(f"  {nombre:<14} {'FALTA en la salida de C++':>38}")
            return 1
        a = np.asarray(ref[nombre], dtype=np.float64).ravel()
        b = np.asarray(cpp[nombre], dtype=np.float64).ravel()
        if a.shape != b.shape:
            print(f"  {nombre:<14} formas distintas: {a.shape} vs {b.shape}")
            return 1
        abso = float(np.max(np.abs(a - b)))
        rel = abso / max(1e-12, float(np.max(np.abs(a))))
        if rel > peor:
            peor, peor_nombre = rel, nombre
        print(f"  {nombre:<14} {abso:>12.3e} {rel:>12.3e}   "
              f"{'ok' if rel <= args.tol else 'REVISAR'}")

    print()
    if peor <= args.tol:
        print(f"RESULTADO: PARIDAD CONFIRMADA (tolerancia {args.tol:g})")
        print(f"  peor error relativo: {peor:.3e} en {peor_nombre}")
        return 0
    print(f"RESULTADO: DISCREPANCIA en {peor_nombre}, error relativo {peor:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
