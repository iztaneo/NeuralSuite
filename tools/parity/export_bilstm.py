"""Exporta un nn.LSTM bidireccional de PyTorch para compararlo con BiLSTM.

La capa de C++ no reimplementa la recurrencia: invierte el eje temporal y
reutiliza la LSTM ya verificada. Eso deja fuera del alcance de las pruebas
internas justo lo que puede salir mal —el orden en que se concatenan las dos
mitades, y si la salida inversa vuelve a alinearse con el tiempo—, porque un
error ahi sigue siendo derivable y consistente consigo mismo. PyTorch sí lo ve.

Correspondencia de parámetros. En C++ el orden lo fija el registro de los
submódulos (`forward`, luego `reverse`), y dentro de cada uno el de la LSTM:

    param000..003  forward.{weight_ih, weight_hh, bias_ih, bias_hh}
    param004..007  reverse.{weight_ih, weight_hh, bias_ih, bias_hh}

que en PyTorch son `*_l0` y `*_l0_reverse`. La salida es [seq, batch, 2*hidden]
con el sentido directo en los primeros `hidden` canales, igual en ambos.

Uso:
    ./venv/bin/python export_bilstm.py --out /tmp/bilstm_ref.nsp
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
    ap.add_argument("--out", default="/tmp/bilstm_ref.nsp")
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--input", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    lstm = nn.LSTM(input_size=args.input, hidden_size=args.hidden, num_layers=1,
                   batch_first=False, bias=True, bidirectional=True)

    x = torch.randn(args.seq, args.batch, args.input, requires_grad=True)

    # Pesos distintos en cada posición y cada canal: con una suma simple, un
    # error que intercambiara las dos mitades de la salida pasaría inadvertido.
    w = torch.zeros(args.seq, args.batch, 2 * args.hidden)
    flat = w.view(-1)
    for i in range(flat.numel()):
        flat[i] = 0.5 + 0.5 * np.cos(1.1 * i)

    out, _ = lstm(x)
    loss = (out * w).sum()
    lstm.zero_grad(set_to_none=False)
    if x.grad is not None:
        x.grad = None
    loss.backward()

    names = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
             "weight_ih_l0_reverse", "weight_hh_l0_reverse",
             "bias_ih_l0_reverse", "bias_hh_l0_reverse"]

    tensors = {
        "meta": np.array([args.seq, args.batch, args.input, args.hidden], dtype=np.float32),
        "x": x.detach().numpy(),
        "w": w.numpy(),
        "ref_out": out.detach().numpy(),
        "ref_loss": np.array([loss.item()], dtype=np.float32),
        "ref_dx": x.grad.detach().numpy(),
        "num_params": np.array([len(names)], dtype=np.float32),
    }
    for i, name in enumerate(names):
        p = getattr(lstm, name)
        tensors[f"param{i:03d}"] = p.detach().numpy()
        tensors[f"grad{i:03d}"] = p.grad.detach().numpy()

    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  seq={args.seq} batch={args.batch} input={args.input} hidden={args.hidden}")
    print(f"  loss ref = {loss.item():.8f}")


if __name__ == "__main__":
    main()
