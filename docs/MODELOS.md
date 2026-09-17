# Los modelos entrenados

Tres modelos de referencia, entrenados íntegramente con NeuralSuite en una CPU.
Cada ficha dice **con qué datos**, **cómo se entrenó**, **qué resultados dio
medidos** y **qué no hace**.

Ninguno se versiona en git: `release/` está excluido. Se reproducen con los
comandos de aquí, o se distribuyen como adjuntos de un
[GitHub Release](https://github.com/iztaneo/NeuralSuite/releases).

**Cada ficha lleva su procedencia**: el commit con el que se entrenó, el hash del
conjunto de datos, la semilla, el build y la máquina. Sin eso, un número medido
no se puede volver a obtener, y en unos meses nadie sabría con qué código salió.

Todos los runs se hicieron en la misma máquina: **Apple M5, 10 núcleos**, macOS
26, **AppleClang 17.0.0**, build `Release` del `Makefile` (`-O3`, sin
`-march=native`), en CPU y con un solo proceso.

---

## 1. `es_base` — modelo de lenguaje en español

| | |
| --- | --- |
| **Qué es** | Un Transformer tipo GPT que genera texto en español, carácter a carácter |
| **Archivos** | `release/es_base.bin` (3.3 MB) y `release/es_base_vocab.txt` (113 caracteres más `<UNK>`, vocabulario de 114) |
| **Tamaño** | ~858 000 parámetros: 4 capas, 4 cabezas, embedding de 128, contexto de 128 |
| **Datos** | 4.9 M caracteres de siete obras de dominio público (Cervantes, Clarín, Pardo Bazán, Unamuno) |
| **Entrenamiento** | 5 000 iteraciones, lote 16, **21.8 minutos** |

**Procedencia**

| | |
| --- | --- |
| Fecha del run | 2026-09-08 |
| Commit vigente | `4d3c286` |
| Datos | `corpus/es/train.txt`, SHA-256 `bb77e27d77cbf7f8…` |
| Semilla | la del programa (el RNG de datos se fija con 1337 en `train_llm`) |
| Checkpoint | `es_base.bin`, SHA-256 `ea0609e93a7954e4…` |
| Vocabulario | `es_base_vocab.txt`, 113 caracteres más `<UNK>` |

### Cómo se entrenó

```bash
./bin/train_llm --data_path corpus/es/train.txt --out_file release/es_base.bin --vocab_file release/es_base_vocab.txt --max_iters 5000 --block_size 128 --n_embd 128 --n_layer 4 --n_head 4 --batch_size 16
```

### Resultados medidos

Con `eval_llm --completo`, sobre las particiones **enteras**:

| Partición | Tokens | Pérdida | Perplejidad |
| --- | --- | --- | --- |
| train | 5 063 680 | 1.7383 | 5.69 |
| val (interior de los mismos cinco libros) | 280 576 | 1.7641 | 5.84 |
| **test: Blasco Ibáñez, autor nunca visto** | 1 568 768 | **1.8292** | **6.23** |

El orden `train < val < test` es el correcto, y la distancia al autor apartado
—0.091 nats— es pequeña. **Generaliza al autor apartado**: la perplejidad pasa de
5.69 en entrenamiento a 6.23 sobre un autor que no se usó para entrenar.

Conviene no estirar esa conclusión. Un autor apartado **no demuestra**
generalización al español en sentido amplio, ni descarta memorización parcial de
fragmentos: demuestra que el modelo no depende de haber visto ese texto.

### Qué hace y qué no

Genera texto con concordancia de artículos, terminaciones verbales, acentos y la
raya de diálogo de la novela española, **con palabras que no existen**. Es lo
esperable de 858 K parámetros a nivel de carácter: aprende **cómo suena** el
idioma, no lo que significa.

- **No responde preguntas ni sigue instrucciones**: no es un modelo de chat.
- **Vocabulario de caracteres**, no BPE: las secuencias son largas.
- **Contexto de 128 caracteres.**

---

## 2. `unet_mnist` — difusión sobre MNIST

| | |
| --- | --- |
| **Qué es** | Un modelo de difusión que **genera dígitos manuscritos nuevos** |
| **Archivos** | `release/unet_mnist.nsf`, `.ema` (965 KB cada uno) y `.opt` |
| **Tamaño** | 246 273 parámetros; U-Net de 32 canales, 1000 pasos de ruido |
| **Datos** | 57 000 imágenes de MNIST (3 000 apartadas para validación) |
| **Entrenamiento** | 80 000 iteraciones, lote 32, **4 h 50 min** |

**Procedencia**

| | |
| --- | --- |
| Fecha del run | 2026-09-12 |
| Commit vigente | `26e9f73`, **más un parche sin commitear** que solo cambiaba el buffer de salida del log, después commiteado como `a817348`. No afecta al cálculo |
| Datos | MNIST, imágenes SHA-256 `ba891046e6505d7a…`, etiquetas `65a50cbbf4e906d7…` |
| Semilla | 7 (por defecto); partición **contigua**, las 3 000 últimas |
| Checkpoint | `unet_mnist.nsf`, SHA-256 `bcf9aca4a092be4e…` |
| Pesos promediados | `.ema`, SHA-256 `bcc60413415f4c0e…` |

### Cómo se entrenó

```bash
caffeinate -i ./bin/train_diffusion --n_imagenes 0 --n_validacion 3000 --canales 32 --pasos 1000 --lote 32 --iteraciones 80000 --calentamiento 1000 --lr 2e-4 --lr_min 1e-5 --ema 0.9995 --evaluar_cada 2000 --muestrear_cada 5000 --guardar_cada 2000 --archivar_cada 10000 --reportar_cada 250 --archivo release/unet_mnist.nsf > logs/difusion_mnist.log 2>&1
```

### Resultados medidos

Generando 64 muestras desde ruido puro, con los pesos promediados (`.ema`):

| | DDPM (1000 pasos) | DDIM (100 pasos) |
| --- | --- | --- |
| Tiempo | 113.6 s | 10.6 s |
| Clases cubiertas | 10 de 10 | 10 de 10 |
| Distancia a la vecina más cercana / distancia media | 0.50 | 0.56 |
| Legibles a ojo | ~45 de 64 | — |

**No hay evidencia de copia directa**, según la distancia a la imagen de
entrenamiento más cercana: el cociente está lejos del 0.15–0.35 que daba el mismo
modelo cuando memorizaba 16 imágenes a propósito. Es una evidencia fuerte, no una
demostración de ausencia de memorización: una medida de distancia en píxeles no
detecta una copia ligeramente desplazada o engrosada.

La brecha de validación se mantuvo plana todo el entrenamiento (+0.0015 a
+0.0019), así que tampoco memoriza el conjunto grande.

### Qué hace y qué no

- **No se le puede pedir un dígito concreto**: no está condicionado por clase.
- **Blanco y negro, 28×28.**
- Una parte de las muestras sale **ambigua**.
- Su checkpoint es anterior al sellado del calendario, así que `sample_diffusion`
  **avisa** de que asume 1000 pasos, que es lo que se usó.

### Reproducir la generación

```bash
./bin/sample_diffusion --muestreador ddpm --n 64 --salida release/muestras_ddpm.png
```

---

## 3. `ae_c4` — autoencoder de la difusión latente

| | |
| --- | --- |
| **Qué es** | El compresor del LDM-1: convierte una imagen 32×32 en un latente 8×8×4 y la reconstruye |
| **Archivos** | `release/ae_c4.nsf` (codificador, 610 KB), `.dec` (decodificador, 492 KB) y `.opt` |
| **Tamaño** | 280 969 parámetros; 32 canales, latente de 4 canales |
| **Datos** | 57 000 imágenes de MNIST rellenadas a 32×32 (3 000 de validación) |
| **Entrenamiento** | 18 000 iteraciones (10 épocas), lote 32, **78 minutos** |

**Procedencia**

| | |
| --- | --- |
| Fecha del run | 2026-09-15 |
| Commit vigente | `ed3ab61` |
| Datos | MNIST, imágenes SHA-256 `ba891046e6505d7a…` |
| Semilla | 7; partición **barajada** con `DataLoader::Partir` |
| Codificador | `ae_c4.nsf`, SHA-256 `8ad31c7ab6687e67…` |
| Decodificador | `ae_c4.nsf.dec`, SHA-256 `07ea6b58233fa226…` |

Los hashes del codificador corresponden al archivo **después de sellar la escala
del latente** con `--medir_escala`: esa orden reescribe los metadatos, así que
cambia el hash sin cambiar un solo peso.

### Cómo se entrenó

```bash
caffeinate -i ./bin/train_autoencoder --n_imagenes 0 --n_validacion 3000 --c_lat 4 --lote 32 --iteraciones 18000 --calentamiento 500 --lr 5e-4 --lr_min 1e-5 --evaluar_cada 1000 --n_eval 1000 --guardar_cada 1000 --archivar_cada 3000 --reportar_cada 250 --archivo release/ae_c4.nsf --png release/ae_c4.png > logs/ae_c4.log 2>&1
```

### Resultados medidos

Sobre 1 000 imágenes de validación, reconstruyendo con la media del latente:

| | PSNR |
| --- | --- |
| **Este modelo (C = 4)** | **33.23 dB** |
| Compresor trivial del mismo tamaño (reducir a 16×16 y ampliar) | 19.46 dB |
| Ventaja | **+13.77 dB** |

El barrido completo, con tres entrenamientos idénticos salvo el latente:

| `C` | Números del latente | Compresión | PSNR validación | Ventaja sobre su referencia |
| --- | --- | --- | --- | --- |
| 1 | 64 | 16× | 29.35 dB | +14.34 dB |
| 2 | 128 | 8× | 30.90 dB | +14.46 dB |
| **4** | 256 | 4× | **33.23 dB** | +13.77 dB |

**Se eligió C = 4 porque difundir cuesta lo mismo con cualquier `C`**: la U-Net
tarda 39.3 ms por paso con C = 1 y 40.3 ms con C = 4, mientras que sobre los
píxeles 32×32 tarda 251.8 ms. Con el mismo coste, conviene la mejor
reconstrucción.

### La escala del latente

Sellada en el checkpoint: media **−0.6936**, sigma **0.6090**, de donde
`escala = 1.6421`. Quien difunda sobre este latente debe usar
**`(z − media) × escala`**, porque la difusión supone datos centrados.

```bash
./bin/train_autoencoder --medir_escala --archivo release/ae_c4.nsf
```

### Qué hace y qué no

- **No genera nada todavía**: comprime y reconstruye. Generar sobre su latente es
  la Fase 19.
- **Pérdida cuadrática**, sin la perceptual ni el discriminador del artículo de
  Rombach, porque la perceptual necesitaría una VGG preentrenada.
- Su reconstrucción de 33.2 dB es el **techo de calidad** de lo que se genere
  sobre él.

---

## Modelos menores

Además hay modelos de OCR en `release/` (`ocr_texto.ns` y variantes),
entrenados con `train_ocr` sobre un corpus sintético. Su medición está en el
[README](../README.md) y en [tools/ocr/README.md](../tools/ocr/README.md): 3.4 %
de error de carácter sobre una página de libro, frente al 0.1 % de Tesseract.

---

## Cómo se comparan honestamente

Tres reglas que se aplicaron a los tres modelos, y que conviene repetir con
cualquier modelo nuevo:

1. **Medir sobre datos que el modelo no vio.** En el LLM, un autor entero
   apartado; en los de imagen, una partición de validación.
2. **Comparar contra un control**, no contra un umbral inventado. Un PSNR alto no
   significa nada si un compresor trivial lo consigue; un dígito bonito no
   significa nada si es una copia del conjunto.
3. **Publicar también el límite.** Cada ficha dice qué no hace el modelo, porque
   es la parte que se olvida al citar un número.
