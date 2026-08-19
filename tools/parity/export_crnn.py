"""Exporta el CRNN de referencia (LLMRasec/src/ocr.py) para compararlo con C++.

Es la comparación que cierra el OCR. Las pruebas internas del CRNN son gruesas
por fuerza: la red es lineal a trozos —tres ReLU y tres pooling cuyo argmax
puede cambiar— y no existe un paso de diferencia finita que evite los codos.
Aquí no hay diferencias finitas: se compara gradiente contra gradiente.

Ejercita además el camino convolucional de extremo a extremo, que hasta ahora
solo tenía comprobación de gradiente. Y ya se vio con `GraphConv` que una
comprobación de gradiente puede pasar sin comprobar lo que dice.

Correspondencia de parámetros. En C++ el orden lo fija el registro de
submódulos —conv1, conv2, conv3, bilstm, fc— y dentro de cada uno el de su capa:

    param000/001  conv1.{weight, bias}   <- cnn[0]
    param002/003  conv2.{weight, bias}   <- cnn[3]
    param004/005  conv3.{weight, bias}   <- cnn[6]
    param006..013 bilstm.{forward, reverse}.{weight_ih, weight_hh, bias_ih, bias_hh}
    param014/015  fc.{weight, bias}      <- fc

`nn.Conv2d` guarda `[out, in, kh, kw]`, igual que `Conv2D`, así que no hay nada
que convertir. `nn.Linear` sí: guarda `[out, in]` y calcula `x·Wᵀ`, mientras que
la capa `Linear` de C++ guarda `[in, out]` y calcula `x·W`.

Uso:
    ./venv/bin/python export_crnn.py --llmrasec ../LLMRasec --out /tmp/crnn_ref.nsp
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity


def load_reference_crnn(llmrasec_dir):
    """Importa `src.ocr` del proyecto PyTorch de referencia."""
    if not os.path.isdir(llmrasec_dir):
        raise SystemExit(
            f"No se encontro el proyecto de referencia en {llmrasec_dir}.\n"
            "Indicalo con --llmrasec."
        )
    sys.path.insert(0, llmrasec_dir)
    from src.ocr import CRNNModel  # noqa: E402

    return CRNNModel


def collect_params_in_cpp_order(model):
    """Devuelve (nombre, parametro, transponer) en el orden de GetParameters()."""
    cnn = model.cnn
    ordered = []
    for cpp_name, idx in (("conv1", 0), ("conv2", 3), ("conv3", 6)):
        ordered.append((f"{cpp_name}.weight", cnn[idx].weight, False))
        ordered.append((f"{cpp_name}.bias", cnn[idx].bias, False))

    lstm = model.lstm
    for cpp_dir, suffix in (("forward", ""), ("reverse", "_reverse")):
        for cpp_name, torch_name in (("weight_ih", "weight_ih_l0"),
                                     ("weight_hh", "weight_hh_l0"),
                                     ("bias_ih", "bias_ih_l0"),
                                     ("bias_hh", "bias_hh_l0")):
            ordered.append((f"bilstm.{cpp_dir}.{cpp_name}",
                            getattr(lstm, torch_name + suffix), False))

    ordered.append(("fc.weight", model.fc.weight, True))
    ordered.append(("fc.bias", model.fc.bias, False))
    return ordered


def run(model, x, w):
    """Forward + backward, devolviendo logits, perdida y gradientes."""
    logits = model(x)
    loss = (logits * w).sum()
    model.zero_grad(set_to_none=False)
    if x.grad is not None:
        x.grad = None
    loss.backward()
    return logits, loss


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_llmrasec = os.path.join(os.path.dirname(repo_root), "LLMRasec")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmrasec", default=default_llmrasec)
    ap.add_argument("--out", default="/tmp/crnn_ref.nsp")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    CRNNModel = load_reference_crnn(args.llmrasec)
    model = CRNNModel(in_channels=1, hidden_dim=args.hidden)
    model.eval()

    height = 32
    x = torch.randn(args.batch, 1, height, args.width, requires_grad=True)

    with torch.no_grad():
        probe = model(x)
    steps, classes = probe.shape[1], probe.shape[2]

    # Pesos distintos en cada posicion: una suma simple no distinguiria un
    # intercambio entre pasos temporales ni entre muestras del lote.
    w = torch.zeros_like(probe)
    flat = w.view(-1)
    for i in range(flat.numel()):
        flat[i] = 0.5 + 0.5 * np.cos(1.1 * i)

    logits, loss = run(model, x, w)
    ordered = collect_params_in_cpp_order(model)

    # El mismo calculo en float64, como referencia de "valor verdadero". Sirve
    # para decidir si una discrepancia con C++ es un defecto o el redondeo que
    # float32 acumula de todos modos: los gradientes de una convolucion son
    # sumas de miles de terminos, y el orden en que se suman no coincide entre
    # las dos implementaciones. Mismo procedimiento que precision_probe.py usa
    # con el GPT.
    torch.manual_seed(args.seed)
    model64 = CRNNModel(in_channels=1, hidden_dim=args.hidden).double()
    model64.eval()
    x64 = x.detach().double().requires_grad_(True)
    _, _ = run(model64, x64, w.double())
    ordered64 = collect_params_in_cpp_order(model64)
    tensors = {
        "meta": np.array([args.batch, height, args.width, args.hidden, steps, classes],
                         dtype=np.float32),
        "x": x.detach().numpy(),
        "w": w.detach().numpy(),
        "ref_out": logits.detach().numpy(),
        "ref_loss": np.array([loss.item()], dtype=np.float32),
        "ref_dx": x.grad.detach().numpy(),
        "num_params": np.array([len(ordered)], dtype=np.float32),
    }
    for i, (_, param, transpose) in enumerate(ordered):
        value, grad = param.detach(), param.grad.detach()
        truth = ordered64[i][1].grad.detach()
        if transpose:
            value, grad, truth = (value.t().contiguous(), grad.t().contiguous(),
                                  truth.t().contiguous())
        tensors[f"param{i:03d}"] = value.numpy()
        tensors[f"grad{i:03d}"] = grad.numpy()
        tensors[f"truth{i:03d}"] = truth.numpy().astype(np.float32)

    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  batch={args.batch} imagen={height}x{args.width} hidden={args.hidden}")
    print(f"  salida = [{args.batch}, {steps}, {classes}]")
    print(f"  loss ref = {loss.item():.8f}")


if __name__ == "__main__":
    main()
