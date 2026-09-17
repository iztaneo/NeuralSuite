#!/usr/bin/env python3
# Copyright 2026 NeuralSuite Authors.
# Licensed under the Apache License, Version 2.0.
"""Recalcula las cifras del proyecto que aparecen en la documentación.

Un documento que dice «26 500 líneas» o «60 pruebas» envejece en silencio: nadie
lo nota hasta que alguien cuenta. Aquí se cuenta sola.

Rellena el bloque delimitado por

    <!-- BEGIN GENERATED STATS -->
    <!-- END GENERATED STATS -->

en `docs/MAPA_DEL_CODIGO.md`, y comprueba que el número de pruebas citado en
cualquier documento coincide con el que hay en `tests/test_suite.cpp`.

Uso:
    python3 tools/docs/update_stats.py              # reescribe
    python3 tools/docs/update_stats.py --comprobar  # falla si está desfasado
"""

import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARPETAS = [
    ("include/", "Interfaces, y la explicación de cada decisión"),
    ("src/", "Implementaciones"),
    ("tests/", "Las {pruebas} pruebas, en un solo archivo"),
    ("apps/", "Los programas de entrenamiento e inferencia"),
    ("tools/", "Paridad con PyTorch y Pillow, corpus y datos"),
    ("demos/", "Una demostración por técnica"),
    ("benchmarks/", "Mediciones de rendimiento"),
]

EXT_CPP = (".h", ".cpp", ".hpp")
EXT_TOOLS = (".py", ".sh", ".h", ".cpp")

INICIO = "<!-- BEGIN GENERATED STATS -->"
FIN = "<!-- END GENERATED STATS -->"

MAPA = os.path.join(RAIZ, "docs", "MAPA_DEL_CODIGO.md")

# El contador de pruebas: cada una se anuncia como "[Test N]" al empezar.
TEST = re.compile(r"\[Test\s+(\d+)\]")
CITA_PRUEBAS = re.compile(r"(\d+)\s+pruebas")


def contar(carpeta, extensiones):
    lineas = archivos = 0
    base = os.path.join(RAIZ, carpeta)
    for raiz, dirs, nombres in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in nombres:
            if not n.endswith(extensiones):
                continue
            archivos += 1
            with open(os.path.join(raiz, n), encoding="utf-8", errors="replace") as f:
                lineas += sum(1 for _ in f)
    return lineas, archivos


def contar_pruebas():
    ruta = os.path.join(RAIZ, "tests", "test_suite.cpp")
    with open(ruta, encoding="utf-8") as f:
        return len(set(TEST.findall(f.read())))


def miles(n):
    return f"{n:,}".replace(",", " ")


def generar():
    pruebas = contar_pruebas()
    filas, total_l, total_a = [], 0, 0
    for carpeta, descripcion in CARPETAS:
        ext = EXT_TOOLS if carpeta == "tools/" else EXT_CPP
        lineas, archivos = contar(carpeta, ext)
        total_l += lineas
        total_a += archivos
        filas.append(f"| `{carpeta}` | {miles(lineas)} | {archivos} | "
                     f"{descripcion.format(pruebas=pruebas)} |")
    cuerpo = [
        INICIO,
        f"NeuralSuite son **{miles(total_l)} líneas** repartidas en "
        f"**{total_a} archivos**, y **{pruebas} pruebas**.",
        "",
        "| Carpeta | Líneas | Archivos | Contenido |",
        "| --- | --- | --- | --- |",
        *filas,
        FIN,
    ]
    return "\n".join(cuerpo), pruebas


def revisar_citas(pruebas, errores):
    """Ningún documento debe citar un número de pruebas distinto del real."""
    for base, dirs, nombres in os.walk(RAIZ):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "build", "build-debug", "bin", "release",
                                "logs", "corpus", "__pycache__", "history"}]
        for n in nombres:
            if not n.endswith(".md"):
                continue
            ruta = os.path.join(base, n)
            with open(ruta, encoding="utf-8") as f:
                for i, linea in enumerate(f, 1):
                    for citado in CITA_PRUEBAS.findall(linea):
                        if int(citado) != pruebas:
                            rel = os.path.relpath(ruta, RAIZ)
                            errores.append(
                                f"{rel}:{i}: dice «{citado} pruebas», hay {pruebas}")


def main():
    comprobar = "--comprobar" in sys.argv
    bloque, pruebas = generar()

    with open(MAPA, encoding="utf-8") as f:
        texto = f.read()

    errores = []
    if INICIO not in texto or FIN not in texto:
        print(f"ERROR: faltan los marcadores {INICIO} / {FIN} en docs/MAPA_DEL_CODIGO.md")
        return 1

    nuevo = re.sub(re.escape(INICIO) + r".*?" + re.escape(FIN), lambda _: bloque,
                   texto, flags=re.S)
    revisar_citas(pruebas, errores)

    if comprobar:
        if nuevo != texto:
            errores.append("docs/MAPA_DEL_CODIGO.md: las cifras estan desfasadas; "
                           "ejecuta python3 tools/docs/update_stats.py")
        if errores:
            print("Estadisticas desfasadas\n")
            for e in errores:
                print("  " + e)
            return 1
        print(f"Estadisticas: al dia ({pruebas} pruebas)")
        return 0

    if nuevo != texto:
        with open(MAPA, "w", encoding="utf-8") as f:
            f.write(nuevo)
        print("docs/MAPA_DEL_CODIGO.md actualizado")
    else:
        print("docs/MAPA_DEL_CODIGO.md ya estaba al dia")
    for e in errores:
        print("  AVISO " + e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
