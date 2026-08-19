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

EXTENSIONES = (".png", ".bmp", ".pgm", ".ppm", ".pbm", ".jpg")

# PNG, BMP y Netpbm reconstruyen los píxeles exactos que se codificaron, así que
# la igualdad tiene que ser byte a byte y una diferencia de uno es un fallo.
#
# JPEG no. La norma no especifica su salida: fija requisitos de precisión para
# la transformada inversa (ITU-T T.83), no un resultado concreto, de modo que
# dos decodificadores correctos difieren en algunos píxeles por una unidad. La
# implementación de referencia usa una transformada entera y aquí se calcula en
# doble precisión, así que las diferencias son inevitables y no son un defecto.
#
# El criterio es la distribución, no el peor píxel, y la razón es que **el
# máximo crece con el tamaño de la imagen**. Cada muestra tiene una probabilidad
# pequeña e independiente de caer en un empate de redondeo donde las dos
# transformadas discrepan en una unidad de más; con más muestras, más
# probabilidad de que alguna lo haga. Se midió: la misma imagen 4:2:0 de 131x96
# da máximo 2 en una versión y 3 en otra —un único píxel de 37728—, con media
# idéntica hasta la cuarta cifra. Poner el umbral en el máximo convertiría eso
# en un fallo y empujaría a relajarlo hasta que pasara, que es justo lo que no
# debe hacerse.
#
# Lo que sí distingue un defecto es la forma del error. Un fallo real —un
# zigzag equivocado, un predictor que no se reinicia, la crominancia sin
# interpolar— no toca un píxel suelto: desplaza la media entera o estropea
# bloques completos. Comprobado mutando el decodificador, esos casos dan medias
# de 3 a 40, entre uno y dos órdenes de magnitud por encima del umbral.
#
# Los números no salen de ajustar hasta que pase. La transformada de este
# proyecto se contrasta contra su definición matemática en `test_suite`, y su
# error propio queda por debajo de 1e-9, así que lo que se mide aquí es
# enteramente el redondeo de la otra implementación.
TOLERANCIA_JPEG_MEDIA = 0.5
TOLERANCIA_JPEG_COLA = 0.001   # fraccion de muestras que puede apartarse mas de 2


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
            continue

        diff = np.abs(got.astype(int) - ref.astype(int))
        if f.endswith(".jpg"):
            cola = (diff > 2).mean()
            ok = diff.mean() <= TOLERANCIA_JPEG_MEDIA and cola <= TOLERANCIA_JPEG_COLA
            marca = "OK   " if ok else "FALLA"
            if not ok:
                fallos.append(name)
            print(f"[{marca}] {name:24s} {w}x{h}x{c}  media={diff.mean():.3f}  "
                  f"cola={cola * 100:.4f}%  max={diff.max()}")
        elif diff.any():
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
    n_jpeg = sum(1 for f in archivos if f.endswith(".jpg"))
    print(f"RESULTADO: los {len(archivos) - n_jpeg} archivos sin perdida se decodifican byte a")
    print(f"byte como Pillow, y los {n_jpeg} JPEG dentro del redondeo que la norma permite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
