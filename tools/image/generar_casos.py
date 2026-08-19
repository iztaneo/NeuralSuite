"""Genera el banco de imágenes con que se compara el decodificador.

Cada archivo se escribe junto a un `.npy` que guarda los píxeles tal y como los
entrega NeuralSuite: la paleta ya expandida a RGB (o RGBA si hay transparencia)
y los 16 bits reducidos a 8. Así `compare_pillow.py` puede comparar byte a byte
sin tener que replicar esas conversiones.

Dos variantes se escriben a mano, byte a byte, porque Pillow no las produce: un
BMP con las filas de arriba abajo (alto negativo) y otro de 16 bits con máscaras
5-6-5. Ambas son corrientes en archivos reales y ambas rompen un decodificador
que dé por supuesto el caso habitual.

Cuidado con el tipo al fabricarlas: `numpy.uint8 << 11` desborda en silencio y
deja el canal rojo a cero. Costó un rato averiguar que el archivo mal formado
era el de la prueba y no lo que leía el decodificador.

Uso:
    ./venv/bin/python generar_casos.py --out /tmp/nsimg
"""

import argparse
import os
import struct

import zlib

import numpy as np
from PIL import Image


def referencia(path, img, indexado=False):
    """Guarda los píxeles como los entrega NeuralSuite.

    `indexado` distingue los formatos cuyos píxeles son índices de paleta —todo
    BMP de 8 bits o menos— de los que son muestras de gris. Nuestro
    decodificador expande siempre la paleta a RGB, porque sus dos colores pueden
    ser cualesquiera, mientras que un PNG bilevel sí es gris de verdad y sale con
    un solo canal. Pillow no hace esa distinción: abre ambos como modo `1`.
    """
    mode = img.mode
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif mode == "1":
        img = img.convert("RGB" if indexado else "L")
    elif mode in ("I;16", "I"):
        np.save(path, (np.array(img).astype(np.uint32) >> 8).astype(np.uint8)[:, :, None])
        return
    a = np.array(img)
    if a.ndim == 2:
        a = a[:, :, None]
    np.save(path, a)


def bmp_a_mano(path, px, top_down=False, bits=24):
    """BMP escrito byte a byte: filas invertidas o 16 bits con máscaras."""
    h, w, _ = px.shape
    stride = ((w * bits + 31) // 32) * 4
    filas = range(h) if top_down else range(h - 1, -1, -1)
    quantized = np.zeros((h, w, 3), dtype=np.uint8)
    body = bytearray()
    for y in filas:
        row = bytearray()
        for x in range(w):
            r, g, b = (int(v) for v in px[y, x])  # int de Python, no uint8
            if bits == 24:
                row += bytes([b, g, r])
                quantized[y, x] = [r, g, b]
            else:
                r5, g6, b5 = r >> 3, g >> 2, b >> 3
                row += struct.pack("<H", (r5 << 11) | (g6 << 5) | b5)
                quantized[y, x] = [r5 * 255 // 31, g6 * 255 // 63, b5 * 255 // 31]
        row += b"\x00" * (stride - len(row))
        body += row

    if bits == 16:
        dib = struct.pack("<IiiHHIIiiII", 56, w, h, 1, 16, 3, len(body), 2835, 2835, 0, 0)
        dib += struct.pack("<IIII", 0xF800, 0x07E0, 0x001F, 0)
    else:
        alto = -h if top_down else h
        dib = struct.pack("<IiiHHIIiiII", 40, w, alto, 1, 24, 0, len(body), 2835, 2835, 0, 0)
    cabecera = b"BM" + struct.pack("<IHHI", 14 + len(dib) + len(body), 0, 0, 14 + len(dib))
    open(path + ".bmp", "wb").write(cabecera + dib + bytes(body))
    np.save(path + ".npy", quantized)


def escribir_png(path, gray, depth=8, interlace=0, filtro=0):
    """Escribe un PNG en gris con el entrelazado, la profundidad y el filtro dados.

    Hace falta un codificador propio porque Pillow no deja elegir ninguna de las
    tres cosas. Lo descubrí por las malas: `img.save(..., interlace=True)` no da
    error, simplemente se ignora, así que los archivos que yo creía entrelazados
    tenían el byte de entrelazado a cero y no probaban Adam7 en absoluto. Lo
    mismo con los filtros: el codificador los elige por su cuenta y nunca llegó a
    emitir Average, de modo que una mutación en ese filtro pasaba desapercibida.
    """
    h, w = gray.shape
    max_value = (1 << depth) - 1
    muestras = (gray.astype(np.uint32) * max_value // 255).astype(np.uint32)

    def empaquetar(fila):
        """Empaqueta una fila de muestras a `depth` bits por muestra."""
        if depth == 8:
            return bytearray(int(v) for v in fila)
        out = bytearray()
        por_byte = 8 // depth
        for i in range(0, len(fila), por_byte):
            byte = 0
            for j in range(por_byte):
                v = int(fila[i + j]) if i + j < len(fila) else 0
                byte |= v << (8 - depth * (j + 1))
            out.append(byte)
        return out

    def filtrar(cruda, previa):
        """Aplica el filtro pedido. Solo se usan los que no dependen del pixel
        anterior ya filtrado, que en gris de 8 bits es el byte contiguo."""
        salida = bytearray([filtro])
        bpp = max(1, depth // 8)
        for i, byte in enumerate(cruda):
            izq = cruda[i - bpp] if i >= bpp else 0
            arr = previa[i] if previa else 0
            esq = previa[i - bpp] if (previa and i >= bpp) else 0
            if filtro == 0:
                v = byte
            elif filtro == 1:
                v = byte - izq
            elif filtro == 2:
                v = byte - arr
            elif filtro == 3:
                v = byte - ((izq + arr) >> 1)
            else:
                p = izq + arr - esq
                pa, pb, pc = abs(p - izq), abs(p - arr), abs(p - esq)
                pred = izq if (pa <= pb and pa <= pc) else (arr if pb <= pc else esq)
                v = byte - pred
            salida.append(v & 0xFF)
        return salida

    ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
             (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    cuerpo = bytearray()
    pasos = ADAM7 if interlace else [(0, 0, 1, 1)]
    for x0, y0, dx, dy in pasos:
        pw = (w - x0 + dx - 1) // dx
        ph = (h - y0 + dy - 1) // dy
        if pw <= 0 or ph <= 0:
            continue
        previa = None
        for fila in range(ph):
            y = y0 + fila * dy
            cruda = empaquetar([muestras[y, x0 + c * dx] for c in range(pw)])
            cuerpo += filtrar(cruda, previa)
            previa = cruda

    def chunk(tipo, datos):
        return (struct.pack(">I", len(datos)) + tipo + datos +
                struct.pack(">I", zlib.crc32(tipo + datos) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, interlace)
    datos = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
             chunk(b"IDAT", zlib.compress(bytes(cuerpo), 6)) + chunk(b"IEND", b""))
    open(path + ".png", "wb").write(datos)
    esperado = (muestras * 255 // max_value).astype(np.uint8)

    # El archivo lo escribo yo y la referencia la calculo yo, asi que si el
    # codificador y el decodificador compartieran un malentendido, coincidirian
    # entre ellos y la prueba no valdria nada. Que Pillow lea el archivo y
    # obtenga lo mismo es lo que garantiza que el PNG es conforme a la norma y
    # no solo conforme a mis suposiciones.
    leido = Image.open(path + ".png")
    if leido.mode == "1":
        leido = leido.convert("L")
    if not np.array_equal(np.array(leido), esperado):
        raise SystemExit(f"{path}.png: Pillow no lee lo que se quiso escribir; "
                         "el codificador de prueba no genera un PNG valido")

    np.save(path + ".npy", esperado[:, :, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nsimg")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    d = args.out
    os.makedirs(d, exist_ok=True)
    np.random.seed(args.seed)

    # Dimensiones no múltiplo de 4 a propósito: destapan errores de relleno de
    # fila en BMP y de empaquetado de bits en PNG.
    W, H = 37, 23
    gray = Image.fromarray(np.random.randint(0, 256, (H, W), dtype=np.uint8))
    rgb = Image.fromarray(np.random.randint(0, 256, (H, W, 3), dtype=np.uint8))
    rgba = Image.fromarray(np.random.randint(0, 256, (H, W, 4), dtype=np.uint8))
    # Degradado: ejercita de verdad los filtros Sub, Up, Average y Paeth de PNG,
    # que sobre ruido puro apenas se distinguen entre sí.
    grad = Image.fromarray(np.add.outer(np.arange(H) * 7, np.arange(W) * 3).astype(np.uint8))
    bw = Image.fromarray((np.random.rand(H, W) > 0.5).astype(np.uint8) * 255).convert("1")

    png = [
        ("png_gris8", gray, {}),
        ("png_rgb8", rgb, {}),
        ("png_rgba8", rgba, {}),
        ("png_degradado", grad, {}),
        ("png_gris_alfa", gray.convert("LA"), {}),
        ("png_paleta", rgb.convert("P", palette=Image.ADAPTIVE, colors=64), {}),
        ("png_sin_comprimir", rgb, {"compress_level": 0}),
        ("png_maxima_compresion", grad, {"compress_level": 9}),
        ("png_bilevel", bw, {}),
        ("png_un_pixel", Image.fromarray(np.array([[200]], dtype=np.uint8)), {}),
        ("png_fila", Image.fromarray(np.random.randint(0, 256, (1, 64), dtype=np.uint8)), {}),
        ("png_columna", Image.fromarray(np.random.randint(0, 256, (64, 1), dtype=np.uint8)), {}),
    ]
    for name, img, kw in png:
        img.save(f"{d}/{name}.png", **kw)
        referencia(f"{d}/{name}.npy", Image.open(f"{d}/{name}.png"))

    Image.fromarray(np.random.randint(0, 65536, (H, W)).astype(np.uint16)).save(f"{d}/png_gris16.png")
    referencia(f"{d}/png_gris16.npy", Image.open(f"{d}/png_gris16.png"))

    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=32)
    pal.save(f"{d}/png_paleta_trns.png", transparency=0)
    referencia(f"{d}/png_paleta_trns.npy", Image.open(f"{d}/png_paleta_trns.png"))

    for name, img in (("bmp24", rgb), ("bmp32", rgb.convert("RGBA")), ("bmp1", bw),
                      ("bmp8pal", gray.convert("P", palette=Image.ADAPTIVE, colors=256))):
        img.save(f"{d}/{name}.bmp")
        referencia(f"{d}/{name}.npy", Image.open(f"{d}/{name}.bmp"), indexado=True)

    # PNG construidos a mano: aqui si se controla lo que se quiere probar.
    g = np.array(grad)
    for filtro, nombre in enumerate(["ninguno", "sub", "up", "average", "paeth"]):
        escribir_png(f"{d}/png_filtro_{nombre}", g, filtro=filtro)
    escribir_png(f"{d}/png_adam7", g, interlace=1)
    escribir_png(f"{d}/png_adam7_paeth", g, interlace=1, filtro=4)
    for depth in (1, 2, 4, 8):
        escribir_png(f"{d}/png_profundidad{depth}", g, depth=depth)
        escribir_png(f"{d}/png_profundidad{depth}_adam7", g, depth=depth, interlace=1)

    px = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    bmp_a_mano(f"{d}/bmp_arriba_abajo", px, top_down=True, bits=24)
    bmp_a_mano(f"{d}/bmp16_565", px, top_down=False, bits=16)

    gray.save(f"{d}/pgm_bin.pgm")
    referencia(f"{d}/pgm_bin.npy", gray)
    rgb.save(f"{d}/ppm_bin.ppm")
    referencia(f"{d}/ppm_bin.npy", rgb)

    g = np.array(gray)
    with open(f"{d}/pgm_txt.pgm", "w") as fh:
        fh.write(f"P2\n# un comentario en medio de la cabecera\n{W} {H}\n255\n")
        fh.write("\n".join(" ".join(str(v) for v in row) for row in g))
    referencia(f"{d}/pgm_txt.npy", gray)

    with open(f"{d}/ppm_txt.ppm", "w") as fh:
        fh.write(f"P3\n{W} {H}\n255\n")
        fh.write("\n".join(" ".join(str(v) for v in px[y].flatten()) for y in range(H)))
    np.save(f"{d}/ppm_txt.npy", px)

    # En PBM el 1 es negro, al reves que en el resto. Y los pixeles se escriben
    # pegados, sin separador: fue el unico defecto real que destapo esta prueba.
    bits = np.random.rand(H, W) > 0.5
    with open(f"{d}/pbm_txt.pbm", "w") as fh:
        fh.write(f"P1\n{W} {H}\n")
        fh.write("\n".join("".join("1" if v else "0" for v in row) for row in bits))
    np.save(f"{d}/pbm_txt.npy", np.where(bits, 0, 255).astype(np.uint8)[:, :, None])

    pbm = Image.fromarray(np.where(bits, 0, 255).astype(np.uint8)).convert("1")
    pbm.save(f"{d}/pbm_bin.pbm")
    np.save(f"{d}/pbm_bin.npy", np.where(bits, 0, 255).astype(np.uint8)[:, :, None])

    n = len([f for f in os.listdir(d) if not f.endswith(".npy")])
    print(f"Escritos {n} archivos en {d}")


if __name__ == "__main__":
    main()
