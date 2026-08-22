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

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Vocabulario: 62 símbolos, el espacio, y una clase de blanco al final.
#
# El blanco es lo que CTC llama *blank* y aquí hacía falta por la misma razón.
# Antes el espacio hacía dos trabajos a la vez: separar palabras y marcar «aquí
# no hay letra», y el decodificado lo descartaba siempre. Con eso, un espacio
# bien predicho se perdía igual y el modelo no podía leer una línea de varias
# palabras aunque acertara todos los caracteres.
#
# Ahora son clases distintas: el blanco se descarta al decodificar y el espacio
# se conserva. El blanco es además lo que se mete entre dos letras iguales para
# que el colapso de repeticiones no las funda.
VOCAB = (string.ascii_uppercase + string.ascii_lowercase + string.digits + " "
         + "áéíóúüñÁÉÍÓÚÜÑ"
         + ".,;:¿?¡!-—()'\"")

# El blanco es la clase que sigue al último símbolo. No forma parte de VOCAB ni
# se escribe en vocab.txt: representarlo obligaría a meter un carácter que no se
# imprime en un archivo de texto, y un NUL ahí es una fuente de sorpresas. Como
# es siempre el último, ambos lados lo deducen del tamaño del vocabulario.
BLANCO = len(VOCAB)
ESPACIO = VOCAB.index(" ")

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
# Ya no se filtra por nombre. Se usan todas las que tengan alfabeto latino
# utilizable, que en un Mac son unas 177 de 120 familias. El filtro por nombre
# dejaba ocho, todas sans-serif, y esa fue una de las causas medidas de que la
# página de la Ilíada —impresa en serif— diera 54% de error de carácter.
PREFERIDAS = []


def dibuja_letras(fuente, minimo_distintos=8):
    """¿La tipografía tiene glifos para todo el vocabulario?

    Cuando a una fuente le falta un carácter dibuja *tofu*: un rectángulo vacío,
    igual para todos. Y el tofu tiene caja delimitadora no nula, así que
    comprobar `getbbox` lo da por bueno.

    Costó caro la primera vez. `ArialHB.ttc` (Arial Hebrew) pasaba el filtro y
    dibujaba tofu para todo el alfabeto latino: como la tipografía se elige al
    azar, **una de cada ocho imágenes del corpus eran rectángulos vacíos
    etiquetados con una palabra real**. No es ruido, es supervisión
    contradictoria, y ponía un techo al entrenamiento hiciera lo que hiciera el
    modelo.

    Al añadir acentos y signos reapareció en su versión sutil: 31 de las 154
    tipografías que pasaban el filtro —fuentes coreanas, bengalíes, de
    símbolos— tienen las latinas básicas y **no** tienen `ñ`, `á` ni `¿`. Con
    ellas dentro, cada palabra acentuada del corpus habría sido medio texto y
    medio caja vacía.

    El criterio es el mismo en los dos casos, y no es comparar contra el glifo
    de reemplazo: se probó y no vale, porque cada fuente sustituye con lo que le
    parece. `Apple Symbols` dibuja `á` y `ñ` con la misma caja de 42 píxeles y
    ninguna de las dos coincide con U+FFFD.

    Lo que sí distingue: **si la fuente tiene los caracteres, cada uno se dibuja
    distinto; si no los tiene, todos salen iguales.** Se exige entonces que los
    acentos y signos produzcan mapas de bits distintos entre sí.
    """
    def mapa(caracter):
        imagen = Image.new("L", (48, 48), 255)
        ImageDraw.Draw(imagen).text((6, 6), caracter, fill=0, font=fuente)
        return imagen.tobytes()

    if len({mapa(c) for c in "AbcQ7zRm5W"}) < minimo_distintos:
        return False

    # Los no ASCII, que son los que faltan en las fuentes de otros alfabetos.
    # Se admite alguna coincidencia: hay tipografías donde dos signos comparten
    # glifo de verdad.
    sondas = "áéíóúñÁÉÍÓÚÑ¿¡—"
    return len({mapa(c) for c in sondas}) >= int(0.8 * len(sondas))


def descubrir_fuentes(maximo=180, por_familia=3):
    """Devuelve rutas de tipografías utilizables, las preferidas primero."""
    candidatas = []
    for base in DIRECTORIOS_FUENTES:
        if not os.path.isdir(base):
            continue
        for patron in ("*.ttf", "*.ttc", "*.otf"):
            candidatas.extend(glob.glob(os.path.join(base, "**", patron), recursive=True))

    def prioridad(ruta):
        return 0

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
        # Se limita cuantas variantes por familia entran: ocho pesos de Arial
        # aportan menos que ocho familias distintas.
        familia = os.path.basename(ruta).split()[0].split(".")[0].lower()
        if sum(1 for f in familias if f == familia) >= por_familia:
            continue
        try:
            fuente = ImageFont.truetype(ruta, 20)
            if not dibuja_letras(fuente):
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

    A un paso que no cae sobre ninguna letra se le asigna la clase de blanco,
    que el decodificado descarta. El espacio entre palabras es una clase
    distinta y sí se conserva: son cosas diferentes y confundirlas impedía leer
    una línea de varias palabras.

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

    clase_blanco = BLANCO
    salida, procedencia = [], []   # procedencia: indice del caracter, o -1
    for paso in range(pasos):
        centro = (paso + 0.5) * ancho_paso
        clase, origen = clase_blanco, -1
        for i, caracter in enumerate(texto):
            if bordes[i] <= centro < bordes[i + 1]:
                clase, origen = indice.get(caracter, clase_blanco), i
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
            salida[paso] = clase_blanco
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

    # Degradación. Un renglón de un libro llega con 14 píxeles de alto y se
    # amplía a 32: el modelo recibe trazos interpolados, no los glifos nítidos
    # con los que se entrenaba. Se reproduce reduciendo y volviendo a ampliar.
    # El desenfoque y el ruido cubren el escaneo y la foto.
    if rng.random() < 0.5:
        factor = rng.uniform(0.35, 0.8)
        pequena = imagen.resize((max(8, int(ancho * factor)), max(8, int(alto * factor))),
                                Image.BILINEAR)
        imagen = pequena.resize((ancho, alto), Image.BILINEAR)
    if rng.random() < 0.3:
        imagen = imagen.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.4:
        pixeles = imagen.load()
        amplitud = rng.randint(4, 20)
        for yy in range(alto):
            for xx in range(ancho):
                v = pixeles[xx, yy] + rng.randint(-amplitud, amplitud)
                pixeles[xx, yy] = max(0, min(255, v))

    return imagen, fuente, x + dx


REALES = ["MITSUBISHI", "MOTORS", "Toyota", "Honda", "Nissan", "Engine", "Speed",
          "Drive", "Japan", "Serie", "Modelo", "Placa", "Folio", "Factura",
          "Total", "Fecha", "Cliente", "Producto", "Cantidad", "Precio",
          "disputa", "entre", "guerra", "madre", "pedirle", "ayude", "los",
          "que", "del", "para", "porque", "hijo", "sobre", "cuando", "desde",
          "primero", "segundo", "tercero", "importe", "unidad", "descuento",
          # Con acentos y eñe. No están por completitud: medido sobre la página
          # de la Ilíada, los caracteres fuera del vocabulario eran el 4.0% de
          # su texto, y sin ejemplos el modelo no puede aprender a emitirlos
          # aunque la clase exista.
          "envía", "envió", "número", "código", "artículo", "días", "está",
          "también", "según", "año", "señor", "diseño", "más", "así", "aquí",
          "después", "quién", "cómo", "página", "línea", "máquina", "última",
          "método", "válido", "teléfono", "dirección", "impresión", "opción"]

# Signos que se pegan a una palabra o cierran un renglón. Un OCR de documentos
# los ve constantemente, y sin ellos en el entrenamiento las comas y los puntos
# de la Ilíada desaparecían de la transcripción.
SIGNOS_FINALES = [",", ".", ";", ":", "?", "!", "..."]
SIGNOS_ENVOLVENTES = [("(", ")"), ("¿", "?"), ("¡", "!"), ("'", "'"), ('"', '"')]

SIMBOLOS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def una_palabra(rng, largo_min=2, largo_max=11):
    """Una palabra real o una cadena aleatoria, a veces con signos."""
    if rng.random() < 0.45:
        palabra = rng.choice(REALES)
    else:
        palabra = "".join(rng.choice(SIMBOLOS)
                          for _ in range(rng.randint(largo_min, largo_max)))

    tirada = rng.random()
    if tirada < 0.12:
        palabra += rng.choice(SIGNOS_FINALES)
    elif tirada < 0.17:
        abre, cierra = rng.choice(SIGNOS_ENVOLVENTES)
        palabra = abre + palabra + cierra
    elif tirada < 0.20:
        # La raya y el guion aparecen entre palabras, no sueltos.
        palabra += rng.choice(["-", "—"])
    return palabra


def renglon(rng, caracteres_min=6, caracteres_max=38):
    """Una línea de varias palabras separadas por espacios.

    Un renglón de libro son unos cincuenta caracteres con espacios, y el modelo
    se entrenaba con palabras sueltas de hasta diez. Medido sobre la página de
    la Ilíada, ese desajuste —junto con la tipografía— era el grueso del 54% de
    error; los acentos, que era lo que se había culpado al principio, son el
    4.1% del texto.

    Se conservan también renglones de una sola palabra: un OCR recibe tanto
    párrafos como campos sueltos de un formulario.
    """
    objetivo = rng.randint(caracteres_min, caracteres_max)
    partes = []
    largo = 0
    while largo < objetivo:
        w = una_palabra(rng)
        partes.append(w)
        largo += len(w) + 1
    return " ".join(partes)


INDICE = {c: i for i, c in enumerate(VOCAB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ocr_datos")
    ap.add_argument("--n", type=int, default=4000, help="imagenes de entrenamiento")
    ap.add_argument("--n-val", type=int, default=400, help="imagenes de validacion")
    # 512 px dan 128 pasos de secuencia, unos cuatro por caracter en un renglon
    # de treinta: la misma densidad con la que ya funcionaba, pero con lineas de
    # la longitud de las de un documento. En 128 px no caben.
    ap.add_argument("--ancho", type=int, default=512)
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
            texto = renglon(rng)
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
