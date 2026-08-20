"""Genera un corpus de líneas de texto renderizadas con tipografías reales.

Escribe imágenes PNG en escala de grises y un archivo de etiquetas, de modo que
el entrenamiento en C++ pueda leerlas con el decodificador de `include/image/`
sin depender de Python. Python interviene una sola vez, para fabricar el corpus:
renderizar una tipografía TrueType exige un intérprete de fuentes, y escribir
uno sería otro proyecto entero.

Está basado en `SynthTextGenerator` de LLMRasec, con dos diferencias que
importan:

- **Las fuentes se buscan también en macOS y Windows.** El original solo mira
  cuatro rutas de Linux; en un Mac no encuentra ninguna y cae a la tipografía
  por defecto de Pillow, que renderiza a 8 píxeles de alto sobre una imagen de
  32. Se comprobó: el texto ocupaba 8 filas de 32 y 55 columnas de 128.
- **El tamaño se ajusta a la caja.** En vez de fijar 18 puntos y confiar, se
  busca el tamaño que llena el alto disponible y, si la palabra no cabe a lo
  ancho, se reduce hasta que quepa. Una palabra recortada es una etiqueta
  mentirosa: el modelo recibe una imagen a la que le faltan letras y una
  etiqueta que las incluye.

Uso:
    ./venv/bin/python generar_dataset.py --out /tmp/ocr_datos --n 4000
"""

import argparse
import glob
import os
import random
import string

from PIL import Image, ImageDraw, ImageFont

# Vocabulario de la implementación de referencia: 62 símbolos más el espacio.
VOCAB = string.ascii_uppercase + string.ascii_lowercase + string.digits + " "

# Directorios de tipografías de los tres sistemas. Se recorren todos: el corpus
# debe poder generarse en la máquina de quien entrena y en la integración
# continua, que corre sobre Linux.
DIRECTORIOS_FUENTES = [
    "/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
    "C:/Windows/Fonts",
]

# Tipografías preferidas, por nombre. Se buscan por coincidencia parcial para no
# depender de la ruta exacta, que cambia entre sistemas y versiones. Son de
# ancho variable y sin adornos, que es lo que aparece en un documento impreso.
PREFERIDAS = ["Helvetica", "Arial", "Verdana", "Tahoma", "DejaVuSans", "LiberationSans",
              "Times New Roman", "Georgia", "Courier", "Menlo", "Consolas", "FreeSans"]


def descubrir_fuentes(maximo=8):
    """Devuelve rutas de tipografías utilizables, las preferidas primero."""
    candidatas = []
    for base in DIRECTORIOS_FUENTES:
        if not os.path.isdir(base):
            continue
        for patron in ("*.ttf", "*.ttc", "*.otf"):
            candidatas.extend(glob.glob(os.path.join(base, "**", patron), recursive=True))

    def prioridad(ruta):
        nombre = os.path.basename(ruta)
        for i, pref in enumerate(PREFERIDAS):
            if pref.lower().replace(" ", "") in nombre.lower().replace(" ", ""):
                return i
        return len(PREFERIDAS)

    # El recorrido recursivo pasa dos veces por los subdirectorios que ademas
    # estan listados aparte, asi que hay que quitar repetidos por ruta real.
    vistas, unicas = set(), []
    for ruta in candidatas:
        real = os.path.realpath(ruta)
        if real not in vistas:
            vistas.add(real)
            unicas.append(ruta)
    candidatas = sorted(unicas, key=lambda r: (prioridad(r), os.path.basename(r)))

    elegidas, familias = [], []
    for ruta in candidatas:
        if prioridad(ruta) >= len(PREFERIDAS):
            continue  # solo las preferidas: el resto son símbolos, emoji o alfabetos ajenos
        # Como mucho dos variantes por familia: ocho pesos distintos de Arial
        # dan menos variedad al modelo que cuatro tipografias diferentes.
        familia = os.path.basename(ruta).split()[0].split(".")[0].lower()
        if sum(1 for f in familias if f == familia) >= 2:
            continue
        try:
            fuente = ImageFont.truetype(ruta, 20)
            # Una tipografía de símbolos puede cargar y no tener letras latinas.
            if fuente.getbbox("Ag")[2] == 0:
                continue
            elegidas.append(ruta)
            familias.append(familia)
        except Exception:
            continue
        if len(elegidas) >= maximo:
            break
    return elegidas


def ajustar_tamano(ruta, texto, ancho, alto, margen=3):
    """Mayor tamaño de fuente con el que `texto` cabe en la caja."""
    mejor = None
    for puntos in range(alto, 5, -1):
        try:
            fuente = ImageFont.truetype(ruta, puntos)
        except Exception:
            continue
        izq, arriba, der, abajo = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
            (0, 0), texto, font=fuente)
        if der - izq <= ancho - 2 * margen and abajo - arriba <= alto - 2 * margen:
            mejor = (fuente, der - izq, abajo - arriba, izq, arriba)
            break
    return mejor


def etiquetas_por_paso(texto, fuente, x0, ancho, pasos, indice):
    """Clase que le corresponde a cada paso de la secuencia que ve el modelo.

    El CRNN devuelve una prediccion por cada cuatro columnas de la imagen. Como
    aqui somos nosotros quienes dibujamos el texto, sabemos en que columnas cae
    cada letra y podemos decir exactamente que le toca a cada paso. Eso convierte
    el problema en una clasificacion normal, con `CrossEntropyLoss`, y evita
    tener que implementar CTC, que es lo que hace falta cuando la alineacion se
    desconoce.

    A un paso que no cae sobre ninguna letra se le asigna el espacio, que ya
    forma parte del vocabulario y que el decodificado descarta.

    Queda un problema, y es el mismo que resuelve CTC con su simbolo en blanco:
    el decodificado colapsa repeticiones consecutivas, asi que dos letras
    iguales seguidas se funden en una y `9XSLxtt` sale como `9XSLxt`. Medido
    sobre dos mil palabras, eso afectaba al 8.9% y era un techo que ningun
    entrenamiento podia superar.

    La solucion es la de CTC sin CTC: como sabemos donde acaba cada letra,
    metemos un espacio en el primer paso de la segunda cuando la anterior es
    igual. Al colapsar, el espacio las separa y las dos sobreviven. Solo puede
    hacerse si esa segunda letra ocupa mas de un paso; si no, se prefiere
    perderla a dejarla sin marcar.
    """
    ancho_paso = ancho / pasos
    # Borde derecho de cada prefijo: la diferencia entre dos consecutivos da la
    # franja horizontal que ocupa cada caracter.
    bordes = [x0]
    for i in range(1, len(texto) + 1):
        bordes.append(x0 + fuente.getlength(texto[:i]))

    clase_espacio = indice[" "]
    salida, procedencia = [], []   # procedencia: indice del caracter, o -1
    for paso in range(pasos):
        centro = (paso + 0.5) * ancho_paso
        clase, origen = clase_espacio, -1
        for i, caracter in enumerate(texto):
            if bordes[i] <= centro < bordes[i + 1]:
                clase, origen = indice.get(caracter, clase_espacio), i
                break
        salida.append(clase)
        procedencia.append(origen)

    # Separar dos letras iguales que se tocan.
    for paso in range(1, pasos):
        anterior, actual = procedencia[paso - 1], procedencia[paso]
        if anterior == -1 or actual == -1 or anterior == actual:
            continue
        if salida[paso - 1] != salida[paso]:
            continue   # letras distintas: el colapso ya las separa
        # Solo si a la segunda le sobra algun paso.
        if sum(1 for o in procedencia if o == actual) > 1:
            salida[paso] = clase_espacio
            procedencia[paso] = -1
    return salida


def renderizar(texto, ruta_fuente, ancho, alto, rng):
    """Dibuja `texto` en una imagen en gris, con tinta oscura sobre fondo claro."""
    ajuste = ajustar_tamano(ruta_fuente, texto, ancho, alto)
    if ajuste is None:
        return None
    fuente, ancho_texto, alto_texto, dx, dy = ajuste

    # Fondo y tinta con contraste variable: un escaneo real no es blanco puro
    # sobre negro puro, y un modelo entrenado solo con los extremos se rompe en
    # cuanto ve un gris.
    fondo = rng.randint(215, 255)
    tinta = rng.randint(0, 70)
    imagen = Image.new("L", (ancho, alto), color=fondo)
    dibujo = ImageDraw.Draw(imagen)

    # Posición centrada, con algo de holgura aleatoria: si el texto cayera
    # siempre en el mismo sitio, el modelo aprenderia esa posicion y no la letra.
    holgura_x = max(0, ancho - ancho_texto)
    holgura_y = max(0, alto - alto_texto)
    x = rng.randint(0, holgura_x) - dx
    y = rng.randint(0, holgura_y) - dy
    dibujo.text((x, y), texto, fill=tinta, font=fuente)
    return imagen, fuente, x + dx


def palabras(rng, n, longitud_min=3, longitud_max=10):
    """Mezcla de palabras reales y cadenas aleatorias del vocabulario."""
    reales = ["MITSUBISHI", "MOTORS", "Toyota", "Honda", "Nissan", "Engine", "Speed",
              "Drive", "Japan", "Serie", "Modelo", "Placa", "Folio", "Factura",
              "Total", "Fecha", "Cliente", "Producto", "Cantidad", "Precio"]
    simbolos = VOCAB.replace(" ", "")
    salida = []
    for _ in range(n):
        if rng.random() < 0.35:
            salida.append(rng.choice(reales))
        else:
            largo = rng.randint(longitud_min, longitud_max)
            salida.append("".join(rng.choice(simbolos) for _ in range(largo)))
    return salida


INDICE = {c: i for i, c in enumerate(VOCAB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ocr_datos")
    ap.add_argument("--n", type=int, default=4000, help="imagenes de entrenamiento")
    ap.add_argument("--n-val", type=int, default=400, help="imagenes de validacion")
    ap.add_argument("--ancho", type=int, default=128)
    ap.add_argument("--alto", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--listar-fuentes", action="store_true")
    args = ap.parse_args()

    fuentes = descubrir_fuentes()
    if not fuentes:
        raise SystemExit("No se encontro ninguna tipografia utilizable en este sistema.")
    print(f"Tipografias encontradas ({len(fuentes)}):")
    for f in fuentes:
        print(f"  {f}")
    if args.listar_fuentes:
        return

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    for particion, cantidad in (("train", args.n), ("val", args.n_val)):
        directorio = os.path.join(args.out, particion)
        os.makedirs(directorio, exist_ok=True)
        etiquetas = []
        descartadas = 0
        pasos = args.ancho // 4  # el CRNN da una prediccion por cada cuatro columnas
        while len(etiquetas) < cantidad:
            texto = palabras(rng, 1)[0]
            resultado = renderizar(texto, rng.choice(fuentes), args.ancho, args.alto, rng)
            if resultado is None:
                descartadas += 1
                continue
            imagen, fuente, x0 = resultado
            por_paso = etiquetas_por_paso(texto, fuente, x0, args.ancho, pasos, INDICE)
            nombre = f"{len(etiquetas):06d}.png"
            imagen.save(os.path.join(directorio, nombre))
            # nombre, texto, y la clase de cada paso separada por espacios
            etiquetas.append(f"{nombre}\t{texto}\t" + " ".join(str(c) for c in por_paso))

        with open(os.path.join(args.out, f"{particion}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(etiquetas) + "\n")
        print(f"{particion}: {len(etiquetas)} imagenes en {directorio} "
              f"({descartadas} descartadas por no caber)")

    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8") as fh:
        fh.write(VOCAB)
    print(f"vocabulario: {len(VOCAB)} simbolos en {args.out}/vocab.txt")


if __name__ == "__main__":
    main()
