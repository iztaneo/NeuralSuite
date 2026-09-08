"""Exporta RMSNorm y SiLU de PyTorch para compararlos con los de C++.

Por qué hace falta, si ya hay gradient check: un gradient check confirma que el
backward deriva el forward **que se escribió**, no que ese forward sea realmente
un RMSNorm. Podría estar restando la media —o sea, ser un LayerNorm— y el
gradient check pasaría igual, porque sería coherente consigo mismo.

`nn.RMSNorm` existe desde PyTorch 2.4. Su `eps` por defecto es None, que se
traduce en el epsilon de la precisión, así que aquí se fija explícitamente al
mismo valor que usa la capa de C++.

Convenciones que coinciden y por eso no hace falta reordenar nada:

- RMSNorm normaliza el **último** eje, igual que la implementación de C++.
- El parámetro se llama `weight` en PyTorch y `gamma` en C++, pero es el mismo
  vector de tamaño [D] y multiplica después de normalizar.
- `nn.SiLU` es exactamente `x * sigmoid(x)`, sin aproximaciones.

Uso:
    ./venv/bin/python export_bloques.py --out /tmp/bloques_ref.nsp
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity

EPS = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bloques_ref.nsp")
    ap.add_argument("--filas", type=int, default=3)
    ap.add_argument("--ancho", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if not hasattr(nn, "RMSNorm"):
        print("ERROR: este PyTorch no trae nn.RMSNorm (hace falta 2.4 o mayor). "
              f"Version instalada: {torch.__version__}", file=sys.stderr)
        return 1

    g = torch.Generator().manual_seed(args.seed)
    N, D = args.filas, args.ancho

    # Entrada y pesos deterministas, para que el binario de C++ reciba
    # exactamente los mismos numeros y la comparacion no dependa del RNG.
    x = torch.randn(N, D, generator=g, dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, D, generator=g, dtype=torch.float32)   # gradiente sembrado
    gamma = torch.randn(D, generator=g, dtype=torch.float32)

    # --- RMSNorm
    rms = nn.RMSNorm(D, eps=EPS, dtype=torch.float32)
    with torch.no_grad():
        rms.weight.copy_(gamma)
    y_rms = rms(x)
    (y_rms * w).sum().backward()
    dx_rms = x.grad.detach().clone()
    dgamma = rms.weight.grad.detach().clone()

    # --- SiLU, sobre la misma entrada
    x2 = x.detach().clone().requires_grad_(True)
    y_silu = nn.functional.silu(x2)
    (y_silu * w).sum().backward()
    dx_silu = x2.grad.detach().clone()

    tensors = {
        "x": x.detach().numpy(),
        "w": w.numpy(),
        "gamma": gamma.numpy(),
        "meta": np.array([N, D], dtype=np.float32),
        "rms_y": y_rms.detach().numpy(),
        "rms_dx": dx_rms.numpy(),
        "rms_dgamma": dgamma.numpy(),
        "silu_y": y_silu.detach().numpy(),
        "silu_dx": dx_silu.numpy(),
    }
    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  filas={N} ancho={D}  torch={torch.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
