#!/usr/bin/env python3
"""Descarga MNIST y lo deja en formato IDX, listo para el lector de C++.

MNIST son 70 000 imágenes de dígitos manuscritos de 28×28, etiquetadas del 0 al
9: 60 000 para entrenar y 10 000 para probar. Es el examen de la Fase 17 —
generar dígitos reconocibles con difusión— porque el resultado se juzga
mirándolo, sin métricas discutibles.

Se descarga del espejo de S3 en vez de yann.lecun.com, que lleva tiempo
bloqueando descargas automáticas.

Los archivos vienen comprimidos con gzip. Se descomprimen aquí, en Python, en
vez de reutilizar el `inflate` del proyecto: ése implementa el envoltorio zlib
que usa PNG, y gzip lleva otra cabecera. Mezclarlos sería una fuente de fallos
sin ninguna ventaja.

Uso:
    python3 tools/data/descargar_mnist.py

Descarga ~11 MB una sola vez y los deja en `corpus/mnist/`. No entrena nada.
"""

import argparse
import gzip
import hashlib
import pathlib
import sys
import urllib.error
import urllib.request

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Los md5 son de los archivos YA DESCOMPRIMIDOS. Comprobarlos es lo que
# distingue "se descargó algo" de "se descargó lo correcto": un espejo puede
# servir un archivo truncado con código 200 y nada fallaría hasta el
# entrenamiento.
ARCHIVOS = [
    ("train-images-idx3-ubyte", "6bbc9ace898e44ae57da46a324031adb", 47040016),
    ("train-labels-idx1-ubyte", "a25bea736e30d166cdddb491f175f624", 60008),
    ("t10k-images-idx3-ubyte", "2646ac647ad5339dbf082846283269ea", 7840016),
    ("t10k-labels-idx1-ubyte", "27ae3e4e09519cfbb04c329615203637", 10008),
]


def descargar(nombre, destino):
    url = BASE + nombre + ".gz"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            comprimido = r.read()
    except urllib.error.URLError as e:
        raise SystemExit(f"ERROR: no se pudo descargar {url}: {e}")
    datos = gzip.decompress(comprimido)
    destino.write_bytes(datos)
    return len(comprimido), len(datos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="corpus/mnist")
    ap.add_argument("--forzar", action="store_true",
                    help="volver a descargar aunque los archivos ya estén")
    args = ap.parse_args()

    salida = pathlib.Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    print(f"  {'archivo':<26} {'bytes':>10} {'md5':>10}")
    fallos = []
    for nombre, md5_esperado, tam_esperado in ARCHIVOS:
        destino = salida / nombre
        if destino.exists() and not args.forzar:
            estado = "en cache"
        else:
            _, n = descargar(nombre, destino)
            estado = "descargado"

        datos = destino.read_bytes()
        md5 = hashlib.md5(datos).hexdigest()
        ok_tam = len(datos) == tam_esperado
        ok_md5 = md5 == md5_esperado
        print(f"  {nombre:<26} {len(datos):>10,} {'ok' if ok_md5 else 'MAL':>10}"
              f"   {estado}")
        if not ok_tam:
            fallos.append(f"{nombre}: {len(datos)} bytes, se esperaban {tam_esperado}")
        if not ok_md5:
            fallos.append(f"{nombre}: md5 {md5}, se esperaba {md5_esperado}")

    print()
    if fallos:
        print("  ❌ VERIFICACIÓN FALLIDA:")
        for f in fallos:
            print(f"     - {f}")
        print("\n  Los archivos NO son los que se esperaban. Un espejo puede servir")
        print("  contenido truncado con código 200, y eso no fallaría hasta el")
        print("  entrenamiento. Se aborta aquí.")
        return 1

    print(f"  ✅ Los cuatro archivos verificados por tamaño y md5, en {salida}/")
    print("     60 000 imágenes de entrenamiento y 10 000 de prueba, 28×28.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
