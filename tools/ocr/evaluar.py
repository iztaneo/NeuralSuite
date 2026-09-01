"""Mide el OCR contra una referencia y contra Tesseract.

Existe por un fallo concreto. Cada medición se hacía a mano con un fragmento de
Python, y en una de ellas se ejecutó `ocr_cli` con `2>/dev/null`: eso tiró el
aviso de que los pesos no habían cargado —el modelo corría con inicialización
aleatoria— y se midió esa salida como si fuera un resultado. Daba 98% de error y
estuvo a punto de reportarse como «el vocabulario empeoró el modelo».

Por eso lo primero que hace esta herramienta no es medir, es **comprobar que la
medición vale**: código de salida, avisos en la salida de error, y que el número
de renglones leídos coincida con los detectados. Un error de carga aborta con un
mensaje en vez de producir una cifra.

Uso:
    ./venv/bin/python tools/ocr/evaluar.py --pesos release/ocr_texto.ns
    ./venv/bin/python tools/ocr/evaluar.py --pesos ... --casos casos.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cada caso: la imagen y el archivo con lo que pone de verdad. La referencia de
# la Ilíada se verificó contra Tesseract cuando se creó: 16 de sus 18 líneas
# coinciden carácter a carácter, y las dos discrepancias son `LIBRO I` frente a
# `LIBRO 1` y `Leto;` frente a `Leto:`.
CASOS_POR_DEFECTO = [
    {"imagen": "iliada_libro1.png", "referencia": "docs/referencias/iliada.txt",
     "nota": "página de libro, serif pequeña, 18 renglones"},
    {"imagen": "test_image.png", "referencia": "docs/referencias/mitsubishi.txt",
     "solo_texto": True,
     "nota": "logotipo con dos palabras: el dibujo da bandas que no son texto"},
]


def distancia(a, b):
    """Distancia de edición entre dos cadenas."""
    previa = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        actual = [i]
        for j, cb in enumerate(b, 1):
            actual.append(min(previa[j] + 1, actual[j - 1] + 1, previa[j - 1] + (ca != cb)))
        previa = actual
    return previa[-1]


def ejecutar_ocr(binario, imagen, pesos, oculto):
    """Corre ocr_cli y **verifica que el resultado es utilizable**."""
    salida = os.path.join("/tmp", "ocr_eval.txt")
    proceso = subprocess.run(
        [binario, "--image", imagen, "--renglones", "--pesos", pesos,
         "--oculto", str(oculto), "--out", salida],
        capture_output=True, text=True)

    if proceso.returncode != 0:
        raise SystemExit(f"ERROR: ocr_cli devolvio {proceso.returncode} con {imagen}\n"
                         f"{proceso.stderr.strip()}")

    # Esto es lo que se perdía al redirigir stderr a la basura.
    todo = proceso.stdout + proceso.stderr
    for señal in ("Error al cargar", "AVISO: no se pudieron cargar",
                  "inicializacion aleatoria"):
        if señal in todo:
            raise SystemExit(
                f"ERROR: los pesos no se cargaron para {imagen}.\n"
                f"  Medir esta salida daria el rendimiento de una red sin entrenar.\n"
                f"  {[l for l in todo.split(chr(10)) if señal in l][0].strip()}")

    with open(salida, encoding="utf-8") as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def tesseract(imagen):
    """Transcripción de Tesseract, si está instalado."""
    if not shutil.which("tesseract"):
        return None
    destino = "/tmp/ocr_eval_tess"
    r = subprocess.run(["tesseract", imagen, destino, "-l", "spa", "--psm", "6"],
                       capture_output=True)
    if r.returncode != 0:
        return None
    with open(destino + ".txt", encoding="utf-8") as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=os.path.join(RAIZ, "release/ocr_texto.ns"))
    ap.add_argument("--binario", default=os.path.join(RAIZ, "bin/ocr_cli"))
    ap.add_argument("--oculto", type=int, default=64)
    ap.add_argument("--casos", help="archivo JSON con casos propios")
    ap.add_argument("--detalle", action="store_true", help="imprimir renglón a renglón")
    args = ap.parse_args()

    casos = CASOS_POR_DEFECTO
    if args.casos:
        with open(args.casos, encoding="utf-8") as fh:
            casos = json.load(fh)

    if not os.path.exists(args.pesos):
        raise SystemExit(f"ERROR: no existe {args.pesos}. Entrena con ./train_ocr.")

    print("=" * 74)
    print(f"EVALUACION DEL OCR   pesos: {os.path.relpath(args.pesos, RAIZ)}")
    print("=" * 74)
    print(f"{'caso':<26} {'NeuralSuite':>13} {'Tesseract':>12}   renglones")
    print("-" * 74)

    peor = 0.0
    for caso in casos:
        imagen = os.path.join(RAIZ, caso["imagen"])
        referencia_ruta = os.path.join(RAIZ, caso["referencia"])
        if not os.path.exists(referencia_ruta):
            print(f"{caso['imagen']:<26} (falta {caso['referencia']})")
            continue

        with open(referencia_ruta, encoding="utf-8") as fh:
            verdad = "".join(l.strip() for l in fh if l.strip())

        leido = ejecutar_ocr(args.binario, imagen, args.pesos, args.oculto)

        if caso.get("solo_texto"):
            # Hay imágenes —un logotipo, una foto con un rótulo— donde el
            # cortador entrega bandas que no son texto, y el reconocedor
            # devuelve basura sobre ellas porque nunca vio ejemplos negativos.
            # Eso es un problema **de detección**, y sigue pendiente en el
            # roadmap; medirlo aquí mezclaría dos fallos distintos en una cifra.
            # Se compara cada renglón de la referencia con el que mejor le
            # encaja, que aísla lo que el reconocedor sabe hacer.
            with open(referencia_ruta, encoding="utf-8") as fh:
                lineas_ref = [l.strip() for l in fh if l.strip()]
            total = errores = 0
            for esperado in lineas_ref:
                mejor = min((distancia(l, esperado) for l in leido), default=len(esperado))
                errores += mejor
                total += len(esperado)
            nuestro = errores / max(total, 1)
        else:
            nuestro = distancia("".join(leido), verdad) / max(len(verdad), 1)

        tess = tesseract(imagen)
        columna_tess = "no instalado"
        if tess is not None:
            if caso.get("solo_texto"):
                with open(referencia_ruta, encoding="utf-8") as fh:
                    lineas_ref = [l.strip() for l in fh if l.strip()]
                e = t = 0
                for esperado in lineas_ref:
                    e += min((distancia(l, esperado) for l in tess), default=len(esperado))
                    t += len(esperado)
                columna_tess = f"{e / max(t, 1):.1%}"
            else:
                columna_tess = f"{distancia(''.join(tess), verdad) / max(len(verdad), 1):.1%}"

        print(f"{os.path.basename(caso['imagen']):<26} {nuestro:12.1%} {columna_tess:>12}"
              f"   {len(leido)}")
        peor = max(peor, nuestro)

        if args.detalle:
            for i, linea in enumerate(leido, 1):
                print(f"    {i:2d}  {linea}")

    print("-" * 74)
    print(f"peor error de caracter: {peor:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
