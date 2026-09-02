#!/usr/bin/env python3
"""Descarga y prepara un corpus de español para entrenar el LLM.

Por qué existe
--------------
El corpus del LLM eran 3287 bytes de Shakespeare en inglés. El modelo tiene
858 880 parámetros: eso son 268 parámetros por token, así que no podía hacer
otra cosa que memorizarlo. Y el resto del proyecto —documentación, OCR con
acentos, Tesseract con `-l spa`— apunta al español.

Por qué Gutenberg y no Wikipedia
--------------------------------
El volcado de Wikipedia en español son 5.2 GB comprimidos para sacar 10 MB. Su
API evita la descarga, pero los artículos al azar son casi todos esbozos: en una
muestra de tres, dos venían vacíos y el tercero tenía 595 bytes. Harían falta
unos 50 000 artículos, la mayoría plantillas de municipios. Repetir «es un
municipio del condado de...» diez mil veces es peor que no tener corpus.

Gutenberg son libros completos de 0.4 a 2.2 MB. Siete archivos dan ~7 MB de
prosa. La contrapartida es real y conviene tenerla presente: es español de los
siglos XVII a XIX, así que un BPE entrenado aquí fusionará formas que hoy nadie
escribe. Sirve para que el modelo aprenda estructura; no para texto moderno.

Las tres particiones
--------------------
En OCR ya pasó que las métricas de validación mejoraban mientras el rendimiento
real empeoraba, porque validación y entrenamiento salían del mismo generador.
Aquí la trampa es idéntica, así que hay tres particiones y no dos:

  entrenamiento / validación : mismos autores. Dice si la optimización avanza.
  prueba                     : **otro autor**, apartado entero. Dice si generaliza.

Blasco Ibáñez no aparece en entrenamiento por eso. Si el modelo escribe bien a
Cervantes pero se hunde con él, la diferencia es generalización y no ruido.

Uso
---
    python3 tools/corpus/preparar_corpus.py
    python3 tools/corpus/preparar_corpus.py --salida corpus/es --forzar

Descarga ~7 MB una sola vez y los deja en caché. No entrena nada.
"""

import argparse
import hashlib
import pathlib
import sys
import unicodedata
import urllib.error
import urllib.request

# Los libros, con el autor que se espera encontrar en la cabecera. El script
# compara lo descargado contra esta expectativa y aborta si no coincide: un
# identificador equivocado traeria otro libro sin que nadie se enterase.
#
# Se evitan las traducciones (Dostoyevski, Hugo, Homero estan en el catalogo en
# español) porque el español traducido tiene otro registro y mezclarlo mediria
# dos cosas a la vez.
ENTRENAMIENTO = [
    (2000,  "Cervantes",   "Don Quijote"),
    (17073, "Alas",        "La Regenta"),
    (52597, "Pardo",       "Insolación y Morriña"),
    (55514, "Pardo",       "Cuentos de amor"),
    (78052, "Unamuno",     "Andanzas y visiones españolas"),
]

# Autor apartado por completo. Nunca entra en entrenamiento ni en validación.
PRUEBA = [
    (23236, "Blasco",      "Mare nostrum"),
    (26983, "Blasco",      "Sangre y arena"),
]

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Gutenberg envuelve cada obra en una cabecera legal y un pie. Hay variantes
# historicas del marcador, asi que se prueban todas y se aborta si no aparece
# ninguna: quedarse con el texto sin recortar meteria licencia en el corpus.
INICIOS = ("*** START OF THE PROJECT GUTENBERG",
           "*** START OF THIS PROJECT GUTENBERG",
           "***START OF THE PROJECT GUTENBERG")
FINALES = ("*** END OF THE PROJECT GUTENBERG",
           "*** END OF THIS PROJECT GUTENBERG",
           "***END OF THE PROJECT GUTENBERG")


def descargar(id_libro, cache):
    """Descarga el libro si no está ya en caché. Devuelve los bytes crudos."""
    destino = cache / f"pg{id_libro}.txt"
    if destino.exists():
        return destino.read_bytes(), False
    url = URL.format(id=id_libro)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            datos = r.read()
    except urllib.error.URLError as e:
        raise SystemExit(f"ERROR: no se pudo descargar {url}: {e}")
    if len(datos) < 50_000:
        raise SystemExit(
            f"ERROR: {url} devolvió {len(datos)} bytes. Un libro no es tan corto; "
            "probablemente sea una página de error.")
    destino.write_bytes(datos)
    return datos, True


def decodificar(datos, id_libro):
    """UTF-8 y, si no, latin-1. Informa de cuál se usó en vez de callarlo."""
    try:
        return datos.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return datos.decode("latin-1"), "latin-1"


def cabecera(texto):
    """Título y autor tal como los declara Gutenberg, para poder verificarlos."""
    titulo = autor = "?"
    for linea in texto[:4000].splitlines():
        if linea.startswith("Title:") and titulo == "?":
            titulo = linea.split(":", 1)[1].strip()
        elif linea.startswith("Author:") and autor == "?":
            autor = linea.split(":", 1)[1].strip()
    return titulo, autor


def recortar(texto, id_libro):
    """Quita la cabecera legal y el pie. Aborta si no encuentra los marcadores."""
    ini = -1
    for marca in INICIOS:
        p = texto.find(marca)
        if p != -1:
            ini = texto.find("\n", p) + 1
            break
    if ini == -1:
        raise SystemExit(
            f"ERROR: en el libro {id_libro} no aparece ningún marcador de inicio. "
            "Recortar a ciegas metería la licencia en el corpus.")

    fin = len(texto)
    for marca in FINALES:
        p = texto.find(marca, ini)
        if p != -1:
            fin = p
            break

    cuerpo = texto[ini:fin]

    # Verificación, no confianza: si el recorte fallo, el cuerpo seguira lleno de
    # menciones a Gutenberg. Alguna suelta es legitima (aparece en portadas y
    # notas), pero decenas significan que quedo texto legal dentro.
    sobras = cuerpo.count("Project Gutenberg")
    if sobras > 5:
        raise SystemExit(
            f"ERROR: tras recortar, el libro {id_libro} conserva {sobras} menciones "
            "a «Project Gutenberg». El recorte no funcionó.")
    return cuerpo


# Restos en ingles que quedan DENTRO de los marcadores y que el recorte no
# alcanza: creditos del transcriptor, la direccion del equipo de correccion y la
# variante antigua del cierre. Son pocas lineas, pero van al principio del libro
# —lo primero que ve el modelo— y no son español.
RESTOS = ("produced by", "proofreading team", "pgdp.net", "transcriber",
          "end of project gutenberg", "end of the project gutenberg",
          "distributed proofread", "updated editions will replace")


def quitar_restos(texto):
    """Quita las líneas de crédito del transcriptor y los cierres antiguos."""
    salida, quitadas = [], 0
    for l in texto.split("\n"):
        bajo = l.lower()
        if any(r in bajo for r in RESTOS):
            quitadas += 1
            continue
        salida.append(l)
    return "\n".join(salida), quitadas


def normalizar(texto):
    """NFC y limpieza de líneas.

    Sin NFC, «é» puede llegar como e + acento combinante: dos puntos de código
    que se ven igual pero son símbolos distintos para el tokenizador, y el
    vocabulario se duplicaría en silencio. Es justo el tipo de error que no se
    ve mirando el archivo.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    lineas = [l.rstrip() for l in texto.split("\n")]

    # Se colapsan los bloques de mas de dos lineas en blanco, que en Gutenberg
    # separan capitulos y no aportan nada al modelo.
    salida, blancos = [], 0
    for l in lineas:
        if l:
            blancos = 0
            salida.append(l)
        else:
            blancos += 1
            if blancos <= 2:
                salida.append(l)
    return "\n".join(salida).strip() + "\n"


def deduplicar(texto):
    """Elimina líneas largas repetidas.

    Solo las largas: una línea corta repetida es diálogo o verso legítimo, y
    borrarla destrozaría la prosa. Las largas que se repiten exactas son
    encabezados de página y avisos.
    """
    vistas, salida, quitadas = set(), [], 0
    for l in texto.split("\n"):
        if len(l) > 40:
            if l in vistas:
                quitadas += 1
                continue
            vistas.add(l)
        salida.append(l)
    return "\n".join(salida), quitadas


def preparar(libros, cache, etiqueta):
    partes, total_bruto = [], 0
    for id_libro, autor_esperado, titulo_esperado in libros:
        datos, nuevo = descargar(id_libro, cache)
        total_bruto += len(datos)
        texto, codif = decodificar(datos, id_libro)
        titulo, autor = cabecera(texto)

        # El identificador es un numero suelto en una lista: si se teclea mal,
        # entra otro libro y el corpus cambia sin aviso. Se comprueba.
        if autor_esperado.lower() not in autor.lower():
            raise SystemExit(
                f"ERROR: el libro {id_libro} dice ser de «{autor}» y se esperaba "
                f"«{autor_esperado}». ¿Identificador equivocado?")

        cuerpo, restos = quitar_restos(recortar(texto, id_libro))
        cuerpo = normalizar(cuerpo)
        partes.append(cuerpo)
        print(f"    #{id_libro:<6} {titulo[:34]:<34} {autor[:24]:<24} "
              f"{len(cuerpo):>9,} car  {codif}  {restos} restos{'  (descargado)' if nuevo else ''}")
    return "\n\n".join(partes), total_bruto


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", default="corpus/es")
    ap.add_argument("--cache", default="corpus/es/.descargas")
    ap.add_argument("--val", type=float, default=0.05,
                    help="fracción de entrenamiento reservada para validación")
    ap.add_argument("--muestra", default="sample_data/es_muestra.txt",
                    help="fragmento pequeño para las pruebas (hoy no se versiona)")
    ap.add_argument("--muestra-kb", type=int, default=64)
    args = ap.parse_args()

    salida = pathlib.Path(args.salida)
    cache = pathlib.Path(args.cache)
    salida.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    print("\n  === ENTRENAMIENTO Y VALIDACIÓN ===")
    texto_tr, bruto_tr = preparar(ENTRENAMIENTO, cache, "entrenamiento")
    print("\n  === PRUEBA (autor apartado, nunca visto en entrenamiento) ===")
    texto_te, bruto_te = preparar(PRUEBA, cache, "prueba")

    texto_tr, quitadas_tr = deduplicar(texto_tr)
    texto_te, quitadas_te = deduplicar(texto_te)

    # El corte de validacion va al final y por lineas completas, no por caracter:
    # partir una palabra por la mitad meteria un fragmento sin sentido en ambos
    # lados y ensuciaria las dos particiones.
    lineas = texto_tr.split("\n")
    corte = int(len(lineas) * (1.0 - args.val))
    entrenamiento = "\n".join(lineas[:corte]).strip() + "\n"
    validacion = "\n".join(lineas[corte:]).strip() + "\n"

    (salida / "train.txt").write_text(entrenamiento, encoding="utf-8")
    (salida / "val.txt").write_text(validacion, encoding="utf-8")
    (salida / "test.txt").write_text(texto_te, encoding="utf-8")

    muestra = pathlib.Path(args.muestra)
    muestra.parent.mkdir(parents=True, exist_ok=True)
    # La muestra sale del principio de validacion, no de entrenamiento: asi un
    # modelo entrenado con este corpus no ha visto el texto de las pruebas.
    muestra.write_text(validacion[:args.muestra_kb * 1024], encoding="utf-8")

    # --- Verificaciones. Ninguna cifra se imprime sin haberla comprobado antes.
    fallos = []
    for nombre, txt in (("train", entrenamiento), ("val", validacion), ("test", texto_te)):
        if unicodedata.normalize("NFC", txt) != txt:
            fallos.append(f"{nombre}: no está en NFC")
        try:
            txt.encode("utf-8").decode("utf-8")
        except UnicodeError:
            fallos.append(f"{nombre}: no es UTF-8 válido")
        # Esta comprobación era «más de 10 menciones» y dio verde teniendo
        # «Produced by ... pgdp.net» como primera línea del conjunto de prueba.
        # Un umbral deja pasar justo lo que es poco y está al principio, que es
        # lo peor: son las primeras líneas que ve el modelo. Ahora es cero.
        for resto in RESTOS:
            if resto in txt.lower():
                fallos.append(f"{nombre}: conserva «{resto}»")
        if "Project Gutenberg" in txt:
            fallos.append(f"{nombre}: conserva menciones a Project Gutenberg")

    # Que las particiones no se solapen es el requisito que hace validas las
    # cifras. Se comprueba con bloques largos, que es donde un solape se notaria.
    bloques_tr = {hashlib.md5(entrenamiento[i:i+400].encode()).hexdigest()
                  for i in range(0, max(len(entrenamiento) - 400, 1), 400)}
    for nombre, txt in (("val", validacion), ("test", texto_te)):
        sol = sum(1 for i in range(0, max(len(txt) - 400, 1), 400)
                  if hashlib.md5(txt[i:i+400].encode()).hexdigest() in bloques_tr)
        if sol:
            fallos.append(f"{nombre}: {sol} bloques aparecen también en entrenamiento")

    car = len(entrenamiento) + len(validacion) + len(texto_te)
    print(f"""
  ========================================================================
  entrenamiento : {len(entrenamiento):>10,} caracteres   {salida/'train.txt'}
  validación    : {len(validacion):>10,} caracteres   {salida/'val.txt'}
  prueba        : {len(texto_te):>10,} caracteres   {salida/'test.txt'}
  muestra       : {muestra.stat().st_size:>10,} bytes        {muestra}

  descargado    : {(bruto_tr + bruto_te)/1e6:.1f} MB      líneas duplicadas quitadas: {quitadas_tr + quitadas_te}
  símbolos únicos: {len(set(entrenamiento))}
  tokens/parámetro con el modelo actual (858 880 par.): {car/858880:.1f}
  ========================================================================""")

    if fallos:
        print("\n  ❌ VERIFICACIÓN FALLIDA:")
        for f in fallos:
            print(f"     - {f}")
        return 1
    print("\n  ✅ Verificado: NFC, UTF-8, sin texto legal y sin solape entre particiones.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
