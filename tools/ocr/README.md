# Corpus de entrenamiento para el OCR

Genera líneas de texto renderizadas con tipografías reales, como imágenes PNG
más un archivo de etiquetas. El entrenamiento (`train_ocr`) las lee con el
decodificador de `include/image/`, así que **entrenar no necesita Python**:
Python interviene una sola vez, para fabricar el corpus.

```bash
../LLMRasec/venv/bin/python tools/ocr/generar_dataset.py --out /tmp/ocr_datos --n 4000
make && ./bin/train_ocr --datos /tmp/ocr_datos --epocas 30
```

Podría no ser Python, pero renderizar una tipografía TrueType exige un
intérprete de fuentes completo —contornos, *hinting*, espaciado entre pares— y
eso es otro proyecto, no una utilidad.

## Qué cambia respecto al generador de la referencia

Está basado en `SynthTextGenerator` de `LLMRasec/src/dataset_generator.py`, que
no se toca: es el oráculo contra el que se comprueba NeuralSuite y debe quedar
tal cual. Tres diferencias:

**Las tipografías se buscan en los tres sistemas.** El original mira cuatro
rutas de Linux. En un Mac no encuentra ninguna y cae a la tipografía por defecto
de Pillow — sin dar ningún error. Se comprobó qué producía: el texto ocupaba
**8 filas de 32 y 55 columnas de 128**. Habría entrenado, desperdiciando tres
cuartas partes de la imagen, y el síntoma habría sido «el modelo aprende mal»
en vez de «los datos están mal».

**El tamaño se ajusta a la caja.** En vez de fijar 18 puntos, se busca el mayor
que quepa. Si la palabra no cabe ni al mínimo, se descarta: una palabra recortada
es una etiqueta mentirosa — la imagen pierde letras y la etiqueta las conserva.

**Se emite la etiqueta de cada paso de la secuencia**, que es lo que permite
entrenar sin CTC.

## Por qué no hace falta CTC aquí

El CRNN devuelve una predicción por cada cuatro columnas: 32 pasos para una
imagen de 128 px. Entrenar eso normalmente exige CTC, porque no se sabe qué
letra corresponde a cada paso.

Aquí sí se sabe: las imágenes las dibujamos nosotros, y `font.getlength()` da el
borde de cada prefijo, o sea la franja horizontal que ocupa cada carácter. Con
eso cada paso tiene su clase y basta `CrossEntropyLoss`, que ya está verificado
contra PyTorch. Se evitan unas trescientas líneas de pérdida nueva sin comprobar.

**Lo que se pierde:** no se puede entrenar con un corpus ajeno ya etiquetado,
donde la alineación se desconoce. Para eso sí haría falta CTC.

## El techo del decodificado, y cómo se levantó

El decodificado colapsa repeticiones consecutivas, así que dos letras iguales
seguidas se funden: `9XSLxtt` se lee `9XSLxt`. Eso no lo arregla entrenar más;
es un techo. Medido sobre 2000 palabras, reconstruyendo el texto a partir de las
propias etiquetas:

| | Antes | Después |
| --- | --- | --- |
| Palabras reconstruidas exactamente | 91.0% | **99.9%** |
| Fallan por letra repetida | 8.9% | 0.1% |
| Fallan por otra causa | 0% | 0% |

El «después» aplica la idea de CTC sin implementarlo: como conocemos la
alineación, se marca como espacio el primer paso de una letra cuando la anterior
es igual. Al colapsar, el espacio las separa y las dos sobreviven. Solo puede
hacerse si a esa segunda letra le sobra algún paso; el 0.1% restante son `II` e
`ii`, tan estrechas que ocupan uno solo.

Que el 0% falle «por otra causa» es la comprobación de que el cálculo de la
alineación es correcto: si estuviera desplazado, aparecerían fallos que no se
explican por letras repetidas.

## Formato

```
train.txt   nombre.png <TAB> texto <TAB> clase_paso0 clase_paso1 ... clase_paso31
vocab.txt   los 63 símbolos, en orden de índice
train/      las imágenes PNG en escala de grises
val/        igual, para validación
```
