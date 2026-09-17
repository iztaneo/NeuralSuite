#!/usr/bin/env python3
"""Comprueba que la documentación no miente sobre el árbol de archivos.

Existe por un caso real: dos planes ya ejecutados siguieron anunciando «RoPE
pendiente» durante meses, y varios documentos citaban rutas que se habían
renombrado. Nada fallaba, porque un documento no compila. Esto lo compila.

Tres comprobaciones:

1. **Enlaces rotos**: todo enlace relativo de un `.md` apunta a algo que existe.
2. **Rutas de código inventadas**: todo lo que parezca una ruta del proyecto
   dentro de comillas invertidas (`include/...`, `src/...`, `tools/...`) existe.
3. **Documentos huérfanos**: todo `.md` de `docs/` está enlazado desde algún
   sitio, para que nadie escriba una guía que nadie encuentra.

Uso:
    python3 tools/docs/comprobar_docs.py
"""

import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Carpetas que no son documentación del proyecto.
EXCLUIR = {".git", "build", "build-debug", "bin", "release", "logs", "corpus",
           "node_modules", "__pycache__"}

# Prefijos que delatan una ruta del repositorio dentro de comillas invertidas.
PREFIJOS_CODIGO = ("include/", "src/", "tools/", "apps/", "tests/", "demos/",
                   "docs/", ".github/")

# Los planes ya ejecutados están congelados y citan rutas del plan original, que
# era en Python. Se comprueban sus enlaces, pero no sus rutas de código.
SIN_RUTAS = ("docs/history/planes_antiguos",)

ENLACE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODIGO = re.compile(r"`([^`\n]+)`")
BLOQUE = re.compile(r"```.*?```", re.S)


def sin_bloques(texto):
    """Quita los bloques de código: un lambda de C++ parece un enlace roto."""
    return BLOQUE.sub("", texto)


def markdowns():
    for base, dirs, archivos in os.walk(RAIZ):
        dirs[:] = [d for d in dirs if d not in EXCLUIR]
        for a in archivos:
            if a.endswith(".md"):
                yield os.path.join(base, a)


def revisar_enlaces(ruta, texto, errores, enlazados):
    carpeta = os.path.dirname(ruta)
    for destino in ENLACE.findall(texto):
        destino = destino.split()[0]  # descarta el título opcional del enlace
        if destino.startswith(("http://", "https://", "mailto:", "#")):
            continue
        limpio = destino.split("#")[0]
        if not limpio:
            continue
        absoluto = os.path.normpath(os.path.join(carpeta, limpio))
        if limpio.endswith(".md"):
            enlazados.add(absoluto)
        if not os.path.exists(absoluto):
            rel = os.path.relpath(ruta, RAIZ)
            errores.append(f"{rel}: enlace roto -> {destino}")


def revisar_rutas_de_codigo(ruta, texto, errores):
    rel = os.path.relpath(ruta, RAIZ)
    if rel.replace(os.sep, "/").startswith(SIN_RUTAS):
        return
    for linea in texto.splitlines():
        # LLMRasec es el proyecto vecino que hace de oráculo: sus rutas no están
        # en este árbol y citarlas es correcto.
        if "LLMRasec" in linea:
            continue
        revisar_linea(rel, linea, errores)


def revisar_linea(rel, texto, errores):
    for fragmento in CODIGO.findall(texto):
        candidato = fragmento.strip().split()[0].rstrip(".,;:)")
        if not candidato.startswith(PREFIJOS_CODIGO):
            continue
        if not os.path.exists(os.path.join(RAIZ, candidato)):
            errores.append(f"{rel}: ruta inexistente -> {candidato}")


def main():
    errores = []
    enlazados = set()
    textos = {}
    for ruta in markdowns():
        with open(ruta, encoding="utf-8") as f:
            textos[ruta] = f.read()
    for ruta, texto in textos.items():
        limpio = sin_bloques(texto)
        revisar_enlaces(ruta, limpio, errores, enlazados)
        revisar_rutas_de_codigo(ruta, limpio, errores)

    # Huérfanos: un .md de docs/ que nadie enlaza. README.md de cada carpeta no
    # cuenta, porque es la portada y se encuentra sola.
    for ruta in textos:
        rel = os.path.relpath(ruta, RAIZ)
        if not rel.startswith("docs" + os.sep):
            continue
        if os.path.basename(ruta) == "README.md":
            continue
        if os.path.normpath(ruta) not in enlazados:
            errores.append(f"{rel}: documento huerfano, nadie lo enlaza")

    if errores:
        print(f"Documentacion: {len(errores)} problema(s)\n")
        for e in sorted(errores):
            print("  " + e)
        return 1
    print(f"Documentacion: {len(textos)} archivos .md revisados, todo coherente")
    return 0


if __name__ == "__main__":
    sys.exit(main())
