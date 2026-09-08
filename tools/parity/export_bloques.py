"""Exporta RMSNorm, SiLU y GroupNorm de PyTorch para compararlos con los de C++.

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

    # --- GroupNorm, sobre un tensor [N, C, H, W] aparte.
    #
    # Ojo con la asimetria que es facil equivocar: las estadisticas son POR
    # GRUPO —sobre (C/G, H, W)— pero gamma y beta son POR CANAL. Una
    # implementacion que aplicara gamma por grupo daria numeros plausibles.
    N4, C4, H4, W4, G4 = 2, 6, 3, 4, 3
    xg = torch.randn(N4, C4, H4, W4, generator=g, dtype=torch.float32,
                     requires_grad=True)
    wg = torch.randn(N4, C4, H4, W4, generator=g, dtype=torch.float32)
    gn = nn.GroupNorm(G4, C4, eps=EPS, dtype=torch.float32)
    with torch.no_grad():
        gn.weight.copy_(torch.randn(C4, generator=g, dtype=torch.float32))
        gn.bias.copy_(torch.randn(C4, generator=g, dtype=torch.float32))
    y_gn = gn(xg)
    (y_gn * wg).sum().backward()

    tensors = {
        "gn_x": xg.detach().numpy(),
        "gn_w": wg.numpy(),
        "gn_gamma": gn.weight.detach().numpy(),
        "gn_beta": gn.bias.detach().numpy(),
        "gn_meta": np.array([N4, C4, H4, W4, G4], dtype=np.float32),
        "gn_y": y_gn.detach().numpy(),
        "gn_dx": xg.grad.detach().numpy(),
        "gn_dgamma": gn.weight.grad.detach().numpy(),
        "gn_dbeta": gn.bias.grad.detach().numpy(),
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
