"""Compara el decodificador de imagen de NeuralSuite contra Pillow, byte a byte.

Es el mismo planteamiento que `tools/parity` aplica a PyTorch: una
implementación madura como oráculo externo. Aquí el criterio es más estricto que
allí —no hay aritmética en punto flotante de por medio, así que la igualdad debe
ser exacta— y una diferencia de un solo byte es un fallo.

Uso:
    ./venv/bin/python generar_casos.py --out /tmp/nsimg
    ./venv/bin/python compare_pillow.py --dir /tmp/nsimg --tool ./imgdump
"""

import argparse
import os
import subprocess
import sys

import numpy as np

EXTENSIONES = (".png", ".bmp", ".pgm", ".ppm", ".pbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/nsimg")
    ap.add_argument("--tool", required=True, help="ruta del binario imgdump")
    args = ap.parse_args()

    archivos = sorted(f for f in os.listdir(args.dir) if f.endswith(EXTENSIONES))
    if not archivos:
        print(f"No hay imagenes en {args.dir}. Ejecuta antes generar_casos.py.")
        return 1

    print("=" * 68)
    print("DECODIFICACION DE IMAGEN: NeuralSuite frente a Pillow")
    print("=" * 68)

    fallos = []
    for f in archivos:
        name = f.rsplit(".", 1)[0]
        volcado = os.path.join(args.dir, name + ".dump")
        r = subprocess.run([args.tool, os.path.join(args.dir, f), volcado],
                           capture_output=True)
        if r.returncode != 0:
            print(f"[FALLA] {name:24s} {r.stderr.decode().strip()}")
            fallos.append(name)
            continue

        raw = open(volcado, "rb").read()
        nl = raw.index(b"\n")
        w, h, c = map(int, raw[:nl].split())
        got = np.frombuffer(raw[nl + 1:], dtype=np.uint8).reshape(h, w, c)
        ref = np.load(os.path.join(args.dir, name + ".npy"))
        if ref.ndim == 2:
            ref = ref[:, :, None]

        if got.shape != ref.shape:
            print(f"[FALLA] {name:24s} forma {got.shape}, Pillow da {ref.shape}")
            fallos.append(name)
        elif not np.array_equal(got, ref):
            diff = np.abs(got.astype(int) - ref.astype(int))
            print(f"[FALLA] {name:24s} {(diff > 0).sum()} bytes distintos, "
                  f"diferencia maxima {diff.max()}")
            fallos.append(name)
        else:
            print(f"[OK   ] {name:24s} {w}x{h}x{c}")

    print("-" * 68)
    if fallos:
        print(f"RESULTADO: {len(fallos)} de {len(archivos)} no coinciden con Pillow")
        print("  " + ", ".join(fallos))
        return 1
    print(f"RESULTADO: los {len(archivos)} archivos se decodifican byte a byte como Pillow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
