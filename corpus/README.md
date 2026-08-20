# Corpus de entrenamiento

Aquí viven los corpus de imágenes que consume `train_ocr`. **No se versionan**:
son datos derivados, pesan decenas de megabytes y git no olvida lo que entra en
su historial. Solo se versiona este archivo.

## Generar uno

```bash
../LLMRasec/venv/bin/python tools/ocr/generar_dataset.py --out corpus/ocr_v1 --n 12000 --n-val 1000
```

## Estructura

```
corpus/ocr_v1/
  train/        12000 PNG en escala de grises, 128x32, una palabra cada uno
  val/           1000 PNG, las mismas características, nunca vistas al entrenar
  train.txt     una línea por imagen: archivo <TAB> texto <TAB> 32 clases
  val.txt       igual
  vocab.txt     los 63 símbolos; la clase de cada uno es su posición
```

El tercer campo de las etiquetas es la clase que le toca a cada uno de los 32
pasos que devuelve el CRNN. Es lo que permite entrenar con `CrossEntropyLoss` en
lugar de CTC: como las imágenes las dibujamos nosotros, se sabe en qué columnas
cae cada letra.

## Reproducibilidad

`--seed` fija la secuencia de palabras, posiciones y contrastes, pero el
generador dibuja con **las tipografías del sistema**. La misma semilla en otra
máquina —o en Linux, donde no hay Helvetica ni Tahoma— produce otras imágenes.
Si un resultado tiene que ser reproducible exactamente, hay que conservar el
corpus, no la semilla.

## Corpus existentes

| | Qué tiene |
| --- | --- |
| `ocr_v1` | 12000 + 1000, vocabulario de 63 símbolos, palabras de 3 a 10 caracteres, ocho tipografías del sistema. El primero limpio de *tofu* (ver el historial de `generar_dataset.py`). |
