"""Exporta RMSNorm, SiLU, GroupNorm y el remuestreo 2D, para comparar con C++.

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

    # --- Upsample2D (vecino mas proximo) y Downsample2D (promedio).
    #
    # Son adjuntas una de otra salvo el factor 1/f^2, asi que se exportan juntas:
    # el gradiente de una es la operacion de la otra. Comparar las dos contra
    # PyTorch cierra las dos mitades de esa simetria.
    Nr, Cr, Hr, Wr = 2, 3, 4, 6
    xu = torch.randn(Nr, Cr, Hr, Wr, generator=g, dtype=torch.float32,
                     requires_grad=True)
    wu = torch.randn(Nr, Cr, Hr * 2, Wr * 2, generator=g, dtype=torch.float32)
    y_up = nn.Upsample(scale_factor=2, mode="nearest")(xu)
    (y_up * wu).sum().backward()

    xd = torch.randn(Nr, Cr, Hr, Wr, generator=g, dtype=torch.float32,
                     requires_grad=True)
    wd = torch.randn(Nr, Cr, Hr // 2, Wr // 2, generator=g, dtype=torch.float32)
    y_dn = nn.AvgPool2d(2)(xd)
    (y_dn * wd).sum().backward()

    # --- CrossAttention contra nn.MultiheadAttention.
    #
    # PyTorch no tiene una CrossAttention de una pieza: se consigue pasandole
    # `query` distinto de `key`/`value`. Tres convenciones suyas hay que
    # desmontar, y las tres son sitios donde es facil equivocarse:
    #
    #  - `batch_first=False` por defecto, o sea [T, B, C]. Se pone en True.
    #  - Empaqueta las tres proyecciones en `in_proj_weight` de [3E, E], en
    #    orden q, k, v.
    #  - `nn.Linear` guarda el peso como [salida, entrada] y el nuestro como
    #    [entrada, salida], asi que TODOS van traspuestos.
    #
    # Como `key` y `value` son el mismo tensor, su `.grad` acumula las dos
    # ramas: es exactamente lo que debe devolver nuestro GradContexto().
    E, Hh, B2, Tq, Tc = 8, 2, 2, 3, 5
    mha = nn.MultiheadAttention(E, Hh, batch_first=True, dtype=torch.float32)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.randn(3 * E, E, generator=g))
        mha.in_proj_bias.copy_(torch.randn(3 * E, generator=g))
        mha.out_proj.weight.copy_(torch.randn(E, E, generator=g))
        mha.out_proj.bias.copy_(torch.randn(E, generator=g))

    qx = torch.randn(B2, Tq, E, generator=g, dtype=torch.float32, requires_grad=True)
    cx = torch.randn(B2, Tc, E, generator=g, dtype=torch.float32, requires_grad=True)
    wx = torch.randn(B2, Tq, E, generator=g, dtype=torch.float32)
    y_ca, _ = mha(qx, cx, cx, need_weights=False)
    (y_ca * wx).sum().backward()

    Wq, Wk, Wv = mha.in_proj_weight.detach().chunk(3, dim=0)
    bq, bk, bv = mha.in_proj_bias.detach().chunk(3, dim=0)

    # --- SwiGLU.
    #
    # PyTorch no tiene un modulo SwiGLU, asi que la referencia se COMPONE con
    # nn.Linear y F.silu. Conviene decirlo: no es lo mismo que contrastar contra
    # nn.RMSNorm, donde la referencia es una pieza que alguien mas escribio.
    # Aqui lo que se valida es que nuestra composicion y su gradiente coincidan
    # con los de PyTorch sobre las mismas operaciones, que sigue siendo util
    # —el autograd de PyTorch deriva la composicion por su cuenta— pero es una
    # garantia mas debil.
    Ds, Hs, Ns = 6, 16, 4
    xs = torch.randn(Ns, Ds, generator=g, dtype=torch.float32, requires_grad=True)
    ws = torch.randn(Ns, Ds, generator=g, dtype=torch.float32)
    lg = nn.Linear(Ds, Hs, dtype=torch.float32)
    lu = nn.Linear(Ds, Hs, dtype=torch.float32)
    ld = nn.Linear(Hs, Ds, dtype=torch.float32)
    with torch.no_grad():
        for capa, (fi, fo) in ((lg, (Ds, Hs)), (lu, (Ds, Hs)), (ld, (Hs, Ds))):
            capa.weight.copy_(torch.randn(fo, fi, generator=g))
            capa.bias.copy_(torch.randn(fo, generator=g))
    y_sw = ld(nn.functional.silu(lg(xs)) * lu(xs))
    (y_sw * ws).sum().backward()

    # --- RoPE.
    #
    # La referencia se escribe aqui con operaciones de torch en vez de tocar
    # LLMRasec, que es el oraculo del proyecto y no se modifica. Para una
    # operacion cerrada como esta basta: la formula esta en el articulo y lo que
    # se contrasta es la aritmetica, no una implementacion ajena.
    #
    # Convencion: pares ADYACENTES (0,1), (2,3), ... que es la del articulo
    # original. LLaMA usa la otra —empareja i con i+hd/2— y no son
    # intercambiables. Aqui se fija la del articulo en los dos lados.
    Br, Tr, Hr2, HDr = 2, 5, 2, 8
    Cr = Hr2 * HDr
    xr = torch.randn(Br, Tr, Cr, generator=g, dtype=torch.float32)
    wr = torch.randn(Br, Tr, Cr, generator=g, dtype=torch.float32)
    POS0 = 3                      # posicion inicial distinta de cero, a proposito

    def rope_ref(v, pos0):
        out = v.clone()
        for t in range(v.shape[1]):
            p = pos0 + t
            for h in range(Hr2):
                o = h * HDr
                for i in range(HDr // 2):
                    th = p / (10000.0 ** (2.0 * i / HDr))
                    c, sn = float(np.cos(th)), float(np.sin(th))
                    a = v[:, t, o + 2 * i].clone()
                    b = v[:, t, o + 2 * i + 1].clone()
                    out[:, t, o + 2 * i] = a * c - b * sn
                    out[:, t, o + 2 * i + 1] = a * sn + b * c
        return out

    y_rope = rope_ref(xr, POS0)
    # El gradiente es la rotacion por el angulo opuesto, o sea la misma
    # operacion con el seno cambiado de signo. Se obtiene rotando con -pos.
    xr_g = xr.clone().requires_grad_(True)
    y_auto = rope_ref(xr_g, POS0)
    (y_auto * wr).sum().backward()

    tensors = {
        "rope_meta": np.array([Br, Tr, Hr2, HDr, POS0], dtype=np.float32),
        "rope_x": xr.numpy(),
        "rope_w": wr.numpy(),
        "rope_y": y_rope.numpy(),
        "rope_dx": xr_g.grad.detach().numpy(),
        "sw_meta": np.array([Ds, Hs, Ns], dtype=np.float32),
        "sw_x": xs.detach().numpy(),
        "sw_w": ws.numpy(),
        "sw_Wg": lg.weight.detach().T.contiguous().numpy(),
        "sw_Wu": lu.weight.detach().T.contiguous().numpy(),
        "sw_Wd": ld.weight.detach().T.contiguous().numpy(),
        "sw_bg": lg.bias.detach().numpy(),
        "sw_bu": lu.bias.detach().numpy(),
        "sw_bd": ld.bias.detach().numpy(),
        "sw_y": y_sw.detach().numpy(),
        "sw_dx": xs.grad.detach().numpy(),
        "sw_dWg": lg.weight.grad.detach().T.contiguous().numpy(),
        "sw_dWu": lu.weight.grad.detach().T.contiguous().numpy(),
        "sw_dWd": ld.weight.grad.detach().T.contiguous().numpy(),
        "ca_meta": np.array([E, Hh, B2, Tq, Tc], dtype=np.float32),
        "ca_q": qx.detach().numpy(),
        "ca_ctx": cx.detach().numpy(),
        "ca_w": wx.numpy(),
        # Traspuestos a la convencion [entrada, salida] de nuestro Linear.
        "ca_Wq": Wq.T.contiguous().numpy(),
        "ca_Wk": Wk.T.contiguous().numpy(),
        "ca_Wv": Wv.T.contiguous().numpy(),
        "ca_Wo": mha.out_proj.weight.detach().T.contiguous().numpy(),
        "ca_bq": bq.numpy(), "ca_bk": bk.numpy(), "ca_bv": bv.numpy(),
        "ca_bo": mha.out_proj.bias.detach().numpy(),
        "ca_y": y_ca.detach().numpy(),
        "ca_dq": qx.grad.detach().numpy(),
        "ca_dctx": cx.grad.detach().numpy(),
        "up_x": xu.detach().numpy(),
        "up_w": wu.numpy(),
        "up_meta": np.array([Nr, Cr, Hr, Wr], dtype=np.float32),
        "up_y": y_up.detach().numpy(),
        "up_dx": xu.grad.detach().numpy(),
        "dn_x": xd.detach().numpy(),
        "dn_w": wd.numpy(),
        "dn_y": y_dn.detach().numpy(),
        "dn_dx": xd.grad.detach().numpy(),
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
