# Comprobación del decodificador de imagen

Contrasta `include/image/` contra **Pillow**, byte a byte. Es el mismo
planteamiento que `tools/parity` aplica a PyTorch —una implementación madura
como oráculo externo— con un criterio más estricto: aquí no hay aritmética en
punto flotante de por medio, así que la igualdad tiene que ser exacta y una
diferencia de un solo byte es un fallo.

```bash
tools/image/run_image_parity.sh ../LLMRasec
```

## Qué se cubre

| | |
| --- | --- |
| PNG | profundidades 1, 2, 4, 8 y 16 bits; los cinco tipos de color; paleta con y sin `tRNS`; los cinco filtros de fila forzados uno a uno; entrelazado Adam7; sin comprimir y a compresión máxima |
| BMP | 1, 4, 8, 16, 24 y 32 bits; paleta; máscaras 5-6-5; filas de abajo arriba y de arriba abajo |
| Netpbm | P1 a P6, las seis variantes, con comentarios en medio de la cabecera |
| Reales | los tres PNG que ya vivían en el repositorio, escritos por herramientas ajenas |

Las dimensiones son 37×23 a propósito. No son múltiplo de 4 ni de 8, que es
justo lo que destapa los errores de relleno de fila en BMP y de empaquetado de
bits en PNG.

## Por qué hay un codificador PNG en `generar_casos.py`

Porque Pillow no deja elegir lo que hace falta probar, y lo hace en silencio.

`img.save(..., interlace=True)` no da error: simplemente se ignora. Los archivos
que este banco llamaba «entrelazados» tenían el byte de entrelazado a cero, de
modo que **Adam7 no se estaba probando en absoluto**. Se descubrió mutando el
decodificador —desplazar un paso de Adam7 no ponía nada en rojo— que es
exactamente como se descubrió, en su día, que la prueba de `GraphConv` no
comprobaba lo que decía.

Lo mismo con los filtros: el codificador los elige fila a fila por su cuenta y
nunca llegaba a emitir Average, así que una mutación en ese filtro pasaba
desapercibida.

El codificador propio resuelve las dos cosas, pero introduce otro problema: si
el codificador y el decodificador compartieran un malentendido, coincidirían
entre ellos y la comparación no valdría nada. Por eso `escribir_png()` abre cada
archivo que genera **con Pillow** y aborta si no lee lo que se quiso escribir.
Pillow sigue siendo el árbitro; lo único que se le quita es la elección de los
parámetros.

## Lo que este banco no puede comprobar

Solo se le dan archivos válidos, así que no dice nada sobre el comportamiento
ante entrada corrupta. Eso está en el test 28 de `test_suite`, que trunca y
corrompe las imágenes incrustadas 1400 veces y exige que cada una se rechace con
un mensaje o se acepte, pero nunca que reviente. Bajo los sanitizadores de la
integración continua, un acceso fuera de rango aborta la ejecución.

Tampoco distingue si el CRC de los trozos se está comprobando: quitar esa
comprobación no cambia el resultado de ningún archivo bien formado. Esa
comprobación concreta vive también en el test 28, alterando el propio campo del
CRC y no los datos —corromper los datos lo detectaría además el Adler-32 de
zlib, y la prueba no distinguiría cuál de los dos actuó.
