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
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsparity

EPS = 1e-5


def referencia_unet(g, EPS):
    """Referencia de UNet2D, en su propia funcion y no suelta en main().

    No es un capricho de estilo. Todos los casos de este exportador comparten un
    unico ambito y el diccionario final se construye al terminar, asi que un
    nombre repetido se lleva el ultimo valor: al escribir esto se reutilizo `xu`,
    que ya usaba el caso de Upsample2D, y `up_x` acabo conteniendo la entrada de
    la U-Net. Ahi se noto porque las formas no cuadraban y el binario aborto; si
    hubieran cuadrado, la comparacion habria pasado midiendo el tensor
    equivocado, que es el peor fallo posible en un arnes de paridad. Dentro de
    una funcion eso no puede ocurrir.
    """
    # --- UNet2D.
    #
    # El ultimo caso de la fase y el mas util: los componentes ya tienen paridad
    # uno a uno, asi que lo que queda por verificar no son los numeros de cada
    # capa sino el **cableado** —el orden de los canales al concatenar, que salto
    # se une con que subida, y que el gradiente que vuelve a cada salto sea la
    # suma de los dos caminos que lo alcanzan. Las diferencias finitas ya
    # comprueban que el backward deriva ESE forward; esto comprueba que ese
    # forward es la arquitectura que dice ser. Son garantias distintas y ninguna
    # sustituye a la otra: un cableado coherente pero equivocado pasa las
    # diferencias finitas sin inmutarse.
    Nu, Cu, DTu, Gu, Hu = 2, 8, 16, 2, 8
    x_unet = torch.randn(Nu, 1, Hu, Hu, generator=g, dtype=torch.float32, requires_grad=True)
    # El forward calcula el embedding a partir de los pasos, no lo recibe hecho,
    # asi que la referencia parte tambien de pasos reales y lo construye igual
    # que TimeEmbedding —senos y luego cosenos, concatenados—, que ya tiene su
    # propia paridad en el caso te_. Pasarle el embedding ya hecho dejaria esa
    # conexion sin comprobar.
    pasos_u = torch.tensor([3.0, 470.0], dtype=torch.float32)
    mit_u = DTu // 2
    fr_u = torch.exp(-math.log(10000.0) * torch.arange(mit_u, dtype=torch.float32) / mit_u)
    ang_u = pasos_u[:, None] * fr_u[None, :]
    t_unet = torch.cat([torch.sin(ang_u), torch.cos(ang_u)], dim=1)
    w_unet = torch.randn(Nu, 1, Hu, Hu, generator=g, dtype=torch.float32)

    def bloque(cin, cout):
        """Un ResBlockTiempo de referencia, ya validado por el caso rb_."""
        d = {"n1": nn.GroupNorm(Gu, cin, eps=EPS, dtype=torch.float32),
             "c1": nn.Conv2d(cin, cout, 3, padding=1, dtype=torch.float32),
             "pt": nn.Linear(DTu, cout, dtype=torch.float32),
             "n2": nn.GroupNorm(Gu, cout, eps=EPS, dtype=torch.float32),
             "c2": nn.Conv2d(cout, cout, 3, padding=1, dtype=torch.float32)}
        # El atajo solo existe cuando cambian los canales; si no, es la identidad.
        if cin != cout:
            d["at"] = nn.Conv2d(cin, cout, 1, dtype=torch.float32)
        with torch.no_grad():
            for k, capa in d.items():
                escala = 1.0 if k in ("n1", "n2") else 0.2
                capa.weight.copy_(torch.randn(capa.weight.shape, generator=g) * escala)
                capa.bias.copy_(torch.randn(capa.bias.shape, generator=g) * escala)
        return d

    def aplicar(d, h, temb):
        z = d["c1"](nn.functional.silu(d["n1"](h)))
        z = z + d["pt"](nn.functional.silu(temb))[:, :, None, None]
        z = d["c2"](nn.functional.silu(d["n2"](z)))
        return z + (d["at"](h) if "at" in d else h)

    u_ce = nn.Conv2d(1, Cu, 3, padding=1, dtype=torch.float32)
    u_cs = nn.Conv2d(Cu, 1, 3, padding=1, dtype=torch.float32)
    u_ns = nn.GroupNorm(Gu, Cu, eps=EPS, dtype=torch.float32)
    with torch.no_grad():
        for capa, escala in ((u_ce, 0.2), (u_cs, 0.2), (u_ns, 1.0)):
            capa.weight.copy_(torch.randn(capa.weight.shape, generator=g) * escala)
            capa.bias.copy_(torch.randn(capa.bias.shape, generator=g) * escala)

    b_b0 = bloque(Cu, Cu)
    b_b1 = bloque(Cu, 2 * Cu)
    b_ce = bloque(2 * Cu, 2 * Cu)
    b_a1 = bloque(4 * Cu, Cu)
    b_a0 = bloque(2 * Cu, Cu)

    h_unet = u_ce(x_unet)
    salto0_unet = aplicar(b_b0, h_unet, t_unet)
    baj0_unet = nn.functional.avg_pool2d(salto0_unet, 2)          # Downsample2D: media 2x2
    salto1_unet = aplicar(b_b1, baj0_unet, t_unet)
    baj1_unet = nn.functional.avg_pool2d(salto1_unet, 2)
    centro_unet = aplicar(b_ce, baj1_unet, t_unet)
    up1_unet = nn.functional.interpolate(centro_unet, scale_factor=2, mode="nearest")
    alt1_unet = aplicar(b_a1, torch.cat([up1_unet, salto1_unet], dim=1), t_unet)   # el que sube primero
    up0_unet = nn.functional.interpolate(alt1_unet, scale_factor=2, mode="nearest")
    alt0_unet = aplicar(b_a0, torch.cat([up0_unet, salto0_unet], dim=1), t_unet)
    y_unet = u_cs(nn.functional.silu(u_ns(alt0_unet)))
    (y_unet * w_unet).sum().backward()

    def vuelca(prefijo, d, destino):
        for k, capa in d.items():
            if isinstance(capa, nn.Linear):
                destino[f"{prefijo}_{k}_w"] = capa.weight.detach().T.contiguous().numpy()
            else:
                destino[f"{prefijo}_{k}_w"] = capa.weight.detach().numpy()
            destino[f"{prefijo}_{k}_b"] = capa.bias.detach().numpy()

    salida_u = {
        "un_meta": np.array([Nu, Cu, DTu, Gu, Hu], dtype=np.float32),
        "un_x": x_unet.detach().numpy(),
        "un_pasos": pasos_u.numpy(),
        "un_w": w_unet.numpy(),
        "un_y": y_unet.detach().numpy(),
        "un_dx": x_unet.grad.detach().numpy(),
        "un_ce_w": u_ce.weight.detach().numpy(), "un_ce_b": u_ce.bias.detach().numpy(),
        "un_cs_w": u_cs.weight.detach().numpy(), "un_cs_b": u_cs.bias.detach().numpy(),
        "un_ns_w": u_ns.weight.detach().numpy(), "un_ns_b": u_ns.bias.detach().numpy(),
    }
    for pre, d in (("un_b0", b_b0), ("un_b1", b_b1), ("un_ct", b_ce),
                   ("un_a1", b_a1), ("un_a0", b_a0)):
        vuelca(pre, d, salida_u)
    return salida_u


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

    # --- DiffusionSchedule.
    #
    # La referencia se construye con operaciones de torch siguiendo el articulo
    # de DDPM. No hay un modulo que importar: el calendario es aritmetica, y lo
    # que se contrasta es esa aritmetica —en particular el producto acumulado,
    # que con 1000 pasos es donde float pierde digitos—.
    PASOS = 1000
    betas_ref = torch.linspace(1e-4, 0.02, PASOS, dtype=torch.float64)
    alphas_ref = 1.0 - betas_ref
    ab_ref = torch.cumprod(alphas_ref, dim=0)

    # q_sample sobre un lote con un paso distinto por ejemplo, que es como se
    # entrena: sortear un t por muestra en vez de recorrerlos todos.
    Nd, Cd, Hd, Wd = 4, 1, 5, 6
    x0d = torch.randn(Nd, Cd, Hd, Wd, generator=g, dtype=torch.float32)
    ruido_d = torch.randn(Nd, Cd, Hd, Wd, generator=g, dtype=torch.float32)
    pasos_d = torch.tensor([0, 7, 500, PASOS - 1], dtype=torch.long)

    ab_sel = ab_ref[pasos_d].to(torch.float32).view(Nd, 1, 1, 1)
    xt_d = torch.sqrt(ab_sel) * x0d + torch.sqrt(1.0 - ab_sel) * ruido_d
    x0_rec = (xt_d - torch.sqrt(1.0 - ab_sel) * ruido_d) / torch.sqrt(ab_sel)

    # --- TimeEmbedding sinusoidal.
    #
    # Convencion de DDPM: primero todos los senos, luego todos los cosenos. La
    # variante intercalada es igual de valida y no coincide con esta, asi que se
    # fija explicitamente en los dos lados.
    DIM_T = 32
    pasos_te = torch.tensor([0.0, 1.0, 7.0, 50.0, 500.0, 999.0], dtype=torch.float32)
    mitad = DIM_T // 2
    frec = torch.exp(-math.log(10000.0) *
                     torch.arange(mitad, dtype=torch.float32) / mitad)
    ang = pasos_te[:, None] * frec[None, :]
    emb_te = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    # --- ResBlockTiempo.
    #
    # No hay modulo que importar: la referencia se COMPONE con nn.GroupNorm,
    # F.silu, nn.Conv2d y nn.Linear siguiendo la estructura de DDPM. Es una
    # garantia mas debil que contrastar contra una pieza ajena —igual que en
    # SwiGLU— pero el autograd de PyTorch deriva la composicion por su cuenta,
    # que es justo lo que aqui interesa: el backward de este bloque tiene cuatro
    # ramas y es donde es facil equivocarse.
    Nb, Cin, Cout, Hb, Wb, DT, G = 2, 4, 6, 8, 8, 16, 2
    xb = torch.randn(Nb, Cin, Hb, Wb, generator=g, dtype=torch.float32,
                     requires_grad=True)
    tb = torch.randn(Nb, DT, generator=g, dtype=torch.float32, requires_grad=True)
    wb = torch.randn(Nb, Cout, Hb, Wb, generator=g, dtype=torch.float32)

    n1 = nn.GroupNorm(G, Cin, eps=EPS, dtype=torch.float32)
    c1 = nn.Conv2d(Cin, Cout, 3, padding=1, dtype=torch.float32)
    pt = nn.Linear(DT, Cout, dtype=torch.float32)
    n2 = nn.GroupNorm(G, Cout, eps=EPS, dtype=torch.float32)
    c2 = nn.Conv2d(Cout, Cout, 3, padding=1, dtype=torch.float32)
    at = nn.Conv2d(Cin, Cout, 1, dtype=torch.float32)
    with torch.no_grad():
        for capa in (n1, n2):
            capa.weight.copy_(torch.randn(capa.weight.shape, generator=g))
            capa.bias.copy_(torch.randn(capa.bias.shape, generator=g))
        for capa in (c1, c2, at, pt):
            capa.weight.copy_(torch.randn(capa.weight.shape, generator=g) * 0.2)
            capa.bias.copy_(torch.randn(capa.bias.shape, generator=g) * 0.2)

    hb = c1(nn.functional.silu(n1(xb)))
    hb = hb + pt(nn.functional.silu(tb))[:, :, None, None]   # por canal
    hb = c2(nn.functional.silu(n2(hb)))
    yb = hb + at(xb)
    (yb * wb).sum().backward()

    unet_t = referencia_unet(g, EPS)

    tensors = {
        "rb_meta": np.array([Nb, Cin, Cout, Hb, Wb, DT, G], dtype=np.float32),
        "rb_x": xb.detach().numpy(),
        "rb_t": tb.detach().numpy(),
        "rb_w": wb.numpy(),
        "rb_n1_g": n1.weight.detach().numpy(), "rb_n1_b": n1.bias.detach().numpy(),
        "rb_n2_g": n2.weight.detach().numpy(), "rb_n2_b": n2.bias.detach().numpy(),
        "rb_c1_w": c1.weight.detach().numpy(), "rb_c1_b": c1.bias.detach().numpy(),
        "rb_c2_w": c2.weight.detach().numpy(), "rb_c2_b": c2.bias.detach().numpy(),
        "rb_at_w": at.weight.detach().numpy(), "rb_at_b": at.bias.detach().numpy(),
        "rb_pt_w": pt.weight.detach().T.contiguous().numpy(),
        "rb_pt_b": pt.bias.detach().numpy(),
        "rb_y": yb.detach().numpy(),
        "rb_dx": xb.grad.detach().numpy(),
        "rb_dt": tb.grad.detach().numpy(),
        "te_meta": np.array([DIM_T, len(pasos_te)], dtype=np.float32),
        "te_pasos": pasos_te.numpy(),
        "te_emb": emb_te.numpy(),
        "dif_meta": np.array([PASOS, Nd, Cd, Hd, Wd], dtype=np.float32),
        "dif_beta": betas_ref.to(torch.float32).numpy(),
        "dif_alpha_bar": ab_ref.to(torch.float32).numpy(),
        "dif_x0": x0d.numpy(),
        "dif_ruido": ruido_d.numpy(),
        "dif_pasos": pasos_d.to(torch.float32).numpy(),
        "dif_xt": xt_d.numpy(),
        "dif_x0_rec": x0_rec.numpy(),
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
    tensors.update(unet_t)
    nsparity.write(args.out, tensors)

    print(f"Escrito {args.out}")
    print(f"  filas={N} ancho={D}  torch={torch.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
