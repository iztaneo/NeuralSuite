"""Exporta un nn.LSTM de PyTorch para compararlo con la capa LSTM de C++.

Esta comparación es la que da sentido a la reimplementación de la capa: un
gradient check solo confirma que el backward deriva el forward que se escribió,
no que ese forward sea realmente un LSTM. Contrastarlo contra `nn.LSTM` sí lo
confirma.

Convenciones que coinciden entre ambas implementaciones, y por eso no hace
falta reordenar ni transponer nada:

- Disposición de pesos: `weight_ih_l0` es [4*hidden, input] y `weight_hh_l0` es
  [4*hidden, hidden], igual que en C++.
- Orden de puertas dentro de esos bloques: i, f, g, o.
- Disposición de la secuencia: [seq_len, batch, input], que es el
  comportamiento por defecto de nn.LSTM (batch_first=False).

Uso:
    ./venv/bin/python export_lstm.py --out /tmp/lstm_ref.nsp
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/lstm_ref.nsp")
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--input", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    lstm = nn.LSTM(input_size=args.input, hidden_size=args.hidden,
                   num_layers=1, batch_first=False, bias=True)

    x = torch.randn(args.seq, args.batch, args.input, requires_grad=True)

    # Pérdida con pesos variables por posición: una suma simple podría ocultar
    # errores que se cancelan entre pasos temporales.
    w = torch.zeros(args.seq, args.batch, args.hidden)
    flat = w.view(-1)
    for i in range(flat.numel()):
        flat[i] = 0.5 + 0.5 * np.cos(1.1 * i)

    out, _ = lstm(x)
    loss = (out * w).sum()
    lstm.zero_grad(set_to_none=False)
    if x.grad is not None:
        x.grad = None
    loss.backward()

    tensors = {
        "meta": np.array([args.seq, args.batch, args.input, args.hidden], dtype=np.float32),
        "x": x.detach().numpy(),
        "w": w.numpy(),
        "ref_out": out.detach().numpy(),
        "ref_loss": np.array([loss.item()], dtype=np.float32),
        "ref_dx": x.grad.detach().numpy(),
        # Orden de GetParameters() en C++: weight_ih, weight_hh, bias_ih, bias_hh
        "param000": lstm.weight_ih_l0.detach().numpy(),
        "param001": lstm.weight_hh_l0.detach().numpy(),
        "param002": lstm.bias_ih_l0.detach().numpy(),
        "param003": lstm.bias_hh_l0.detach().numpy(),
        "grad000": lstm.weight_ih_l0.grad.detach().numpy(),
        "grad001": lstm.weight_hh_l0.grad.detach().numpy(),
        "grad002": lstm.bias_ih_l0.grad.detach().numpy(),
        "grad003": lstm.bias_hh_l0.grad.detach().numpy(),
        "num_params": np.array([4], dtype=np.float32),
    }
    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  seq={args.seq} batch={args.batch} input={args.input} hidden={args.hidden}")
    print(f"  loss ref = {loss.item():.8f}")


if __name__ == "__main__":
    main()
