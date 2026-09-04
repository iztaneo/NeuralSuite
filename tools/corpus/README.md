# Corpus de español

`preparar_corpus.py` descarga siete obras de dominio público de Project
Gutenberg, les quita el envoltorio legal y produce tres particiones.

```bash
python3 tools/corpus/preparar_corpus.py
```

Descarga ~7.2 MB una sola vez y los deja en `corpus/es/.descargas`. Volver a
ejecutarlo no descarga nada. **No entrena nada.**

## Qué produce

| | Caracteres | Contenido |
| --- | --- | --- |
| `corpus/es/train.txt` | 4 938 174 | Cervantes, Clarín, Pardo Bazán, Unamuno |
| `corpus/es/val.txt` | 264 796 | Los mismos autores, 5% final |
| `corpus/es/test.txt` | 1 531 189 | **Blasco Ibáñez**, apartado por completo |
| `sample_data/es_muestra.txt` | 64 KB | Fragmento para las pruebas unitarias |

Las cuatro están en `.gitignore`: son datos derivados y reproducibles, y git no
olvida. La muestra se versionará cuando exista una prueba que la lea; hasta
entonces sería guardar 64 KB permanentes que nadie abre.

## Por qué tres particiones y no dos

En OCR ya ocurrió que las métricas de validación mejoraban mientras el
rendimiento sobre una página real empeoraba, porque validación y entrenamiento
salían del mismo generador. La métrica subía con el modelo sin decir nada del
mundo.

Aquí la trampa es idéntica, así que **el conjunto de prueba es un autor entero
que nunca aparece en entrenamiento**. Si el modelo escribe bien a Cervantes y se
hunde con Blasco Ibáñez, eso es una respuesta, no ruido. Medido con el primer
modelo: perplejidad 5.28 en entrenamiento y **5.90 sobre Blasco Ibáñez**.

### Defecto conocido de `val`

**`val` no es un conjunto de validación**, pese al nombre. Es el 5% final de la
concatenación, y como Unamuno es el último de los cinco libros, `val` es **sólo
Unamuno** —prosa ensayística densa—. No es una muestra de la distribución de
entrenamiento, así que comparar `train` con `val` no mide lo que parece.

Se detectó porque `val` daba peor perplejidad (6.15) que `test` (5.90) siendo las
dos texto no visto, lo cual no tenía sentido hasta mirar qué contenía cada una.

La conclusión que sostiene el modelo —que generaliza— se apoya en `train` frente
a `test`, que sí es válida. Arreglarlo pide repartir la validación entre los
cinco libros en vez de cortar por el final, y **rehacer el corpus obliga a
reentrenar**, así que está anotado y no hecho.

## Por qué Gutenberg y no Wikipedia

El volcado de Wikipedia en español son **5.2 GB comprimidos** para sacar 10 MB.
Su API evita la descarga, pero los artículos al azar son casi todos esbozos: en
una muestra de tres, dos vinieron vacíos y el tercero tenía 595 bytes. Harían
falta unas 50 000 peticiones y la mayoría serían plantillas de municipios.

## La limitación, que es real

Es español de los siglos XVII a XIX. Un BPE entrenado aquí fusionará formas que
hoy nadie escribe. Sirve para que el modelo aprenda estructura del español; no
para texto moderno. Cambiar de corpus más adelante implica rehacer el
vocabulario y, con él, los pesos.

## Verificación

El script comprueba antes de imprimir cifras:

- El autor que declara cada libro coincide con el esperado — un identificador
  mal tecleado traería otra obra sin que nadie se enterase.
- Los marcadores de recorte existen; si no, aborta en vez de recortar a ciegas.
- Cero restos en inglés: créditos del transcriptor, `pgdp.net`, cierres
  antiguos. **Esta comprobación empezó siendo un umbral de «más de 10
  menciones» y dio verde teniendo «Produced by ... pgdp.net» como primera línea
  del conjunto de prueba.** Un umbral deja pasar justo lo que es poco y está al
  principio, que es lo peor. Ahora es cero.
- Todo en NFC. Sin esto, «é» puede llegar como `e` + acento combinante: dos
  puntos de código que se ven igual y que el tokenizador cuenta por separado.
- Las particiones no comparten bloques de 400 caracteres.

Que la verificación muerde está comprobado desactivando la limpieza: el script
falla y enumera los siete restos que encuentra.

## Para qué sirve: cuánto comprimiría un BPE

Ésta era la pregunta que no se podía responder sin corpus. Medido con un BPE de
prueba **entrenado sobre `train` y evaluado sobre `test`** —autor distinto, sin
trampa—, con 150 KB de entrenamiento y 120 KB de evaluación:

| Fusiones | Vocabulario | tok/carácter en test | Frente a bytes |
| --- | --- | --- | --- |
| 400 | ~510 | 0.460 | 2.2× menos tokens |
| 900 | ~1010 | 0.411 | 2.5× |
| 1800 | ~1910 | 0.375 | **2.7×** |

El tokenizador de bytes gasta 1.02 tok/carácter sobre este mismo corpus.

Dos avisos sobre esas cifras. Son con 150 KB de entrenamiento: con los 5 MB
completos mejorarían, porque hay más estadística de la que aprender fusiones.
Y hay un intento anterior que dio 8× de compresión sobre el corpus viejo de
3.2 KB: **era falso**, el BPE memorizaba líneas enteras. Medir sobre texto que
el tokenizador no ha visto es lo que separa una cifra de un espejismo.

Contra eso hay que pesar el coste en parámetros. Sobre el modelo actual
(4 capas, `n_embd` 128, 858 880 parámetros), la tabla de embeddings crece con el
vocabulario:

| Vocabulario | Tabla | Total | Tabla / total |
| --- | --- | --- | --- |
| 256 (hoy) | 32 768 | 858 880 | 3.8% |
| 1024 | 131 072 | 957 184 | 13.7% |
| 4096 | 524 288 | 1 350 400 | 38.8% |
| 8192 | 1 048 576 | 1 874 688 | 55.9% |

A esta escala, un vocabulario de ~1000–2000 es el punto razonable: recorta los
tokens a la mitad larga sin que el diccionario se coma el modelo. Los 50 257 de
GPT-2 corresponden a `n_embd` 768 y 12 capas.

## Añadir libros

Los identificadores están en `ENTRENAMIENTO` y `PRUEBA`, cada uno con el autor
esperado. El catálogo en español tiene unas 920 obras:
`https://gutendex.com/books?languages=es&sort=popular`. Conviene evitar las
traducciones —el catálogo incluye a Dostoyevski y Homero en español— porque el
español traducido tiene otro registro y mezclarlo mediría dos cosas a la vez.
