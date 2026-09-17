# Guía rápida

De cero a ver el framework funcionando. No hace falta instalar nada: ni PyTorch,
ni CUDA, ni BLAS. Solo un compilador de C++17.

Si lo que quieres es **entender** el proyecto en vez de ejecutarlo, empieza por
[MAPA_DEL_CODIGO.md](MAPA_DEL_CODIGO.md).

---

## 1. Compilar

```bash
make
```

Los ejecutables quedan en `bin/`. En Windows, o si prefieres CMake:

```bash
cmake -B build && cmake --build build
```

## 2. Comprobar que está bien

```bash
./bin/test_suite
```

Son **60 pruebas** y tardan **poco más de un segundo** en un portátil. Casi todas
comparan el gradiente analítico contra diferencias finitas: si pasan, el
`Backward` de cada capa deriva de verdad su `Forward`.

Eso es solo la mitad de la verificación. La otra mitad —comparar contra PyTorch,
que es lo que demuestra que la arquitectura es la que dice ser— se explica en
[VERIFICACION.md](VERIFICACION.md) y necesita Python con PyTorch instalado, que
**no hace falta para nada más**.

## 3. Ver una red aprender, sin datos ni descargas

Esto funciona **recién clonado el repositorio**, sin descargar nada:

```bash
./bin/demo_mlp
```

Hay una docena de demos en `bin/demo_*`: perceptrón, MLP, CNN, LSTM, GAN, red
sobre grafos, autograd, difusión. Todas son autocontenidas, generan sus propios
datos y terminan en segundos. El código de cada una está en `demos/`, y está
escrito para leerse.

Lo más cercano a la difusión que funciona sin descargar nada es

```bash
./bin/demo_diffusion
```

que entrena un desruidificador diminuto sobre vectores sintéticos: añade ruido
según el calendario y aprende a predecirlo, que es el núcleo del DDPM y cabe en
una pantalla. **No muestrea**: no genera imágenes. Para eso hacen falta el
modelo entrenado y `sample_diffusion`.

---

## 4. Lo que necesita descargas

El repositorio contiene **el código, no los datos ni los pesos**. Ni el corpus,
ni MNIST, ni los modelos entrenados se versionan: son megabytes derivados, y git
no olvida lo que entra en su historial.

| Para… | Hace falta | Cómo se consigue |
| --- | --- | --- |
| Entrenar el LLM | `corpus/es/train.txt` | `python3 tools/corpus/preparar_corpus.py` (descarga de Project Gutenberg) |
| Entrenar difusión o autoencoder | `corpus/mnist/` | `python3 tools/data/descargar_mnist.py` |
| Generar con los modelos de [MODELOS.md](MODELOS.md) | `release/*.nsf`, `release/es_base.bin` | [Release v0.1.0](https://github.com/iztaneo/NeuralSuite/releases/tag/v0.1.0), 6.3 MB en total |

Los pesos van a `release/`. Comprueba que has bajado lo que crees antes de
usarlos —el release trae los hashes— y ya funcionan los apartados 5 y 6:

```bash
shasum -a 256 -c release/SHA256SUMS.txt
```

## 5. Generar imágenes, con el modelo de difusión entrenado

Con `release/unet_mnist.nsf.ema` en su sitio:

```bash
./bin/sample_diffusion --muestreador ddim --n 64 --salida muestras.png
```

Sale un PNG con 64 dígitos manuscritos **que no existían**. Con DDIM y 100 pasos
tarda unos 10 segundos; con `--muestreador ddpm` y los 1000 pasos, unos dos
minutos y la calidad es algo mejor. Los detalles, en
[GUIA_DIFUSION.md](GUIA_DIFUSION.md).

## 6. Generar texto en español

Con `es_base.bin` y `es_base_vocab.txt` en `release/`:

```bash
./bin/generate_llm --model_path release/es_base.bin --vocab_path release/es_base_vocab.txt --block_size 128 --n_embd 128 --prompt "En un lugar de la Mancha" --max_new_tokens 300 --temperature 0.8
```

Es un modelo de 858 000 parámetros a nivel de carácter: produce prosa con la
cadencia del español y **palabras inventadas**. Eso es lo esperable de su tamaño,
no un defecto; está explicado en [MODELOS.md](MODELOS.md).

---

## 7. Entrenar algo tú mismo

El entrenamiento más corto que da un resultado medible es el modelo de lenguaje:
**5 000 iteraciones en 21.8 minutos** en un Apple M5. Primero el corpus, que
tarda un par de minutos en descargarse y limpiarse:

```bash
python3 tools/corpus/preparar_corpus.py
```

```bash
./bin/train_llm --data_path corpus/es/train.txt --out_file release/mi_modelo.bin --vocab_file release/mi_vocab.txt --max_iters 5000 --block_size 128 --n_embd 128 --n_layer 4 --n_head 4 --batch_size 16
```

Para difusión sobre MNIST hace falta bajar el conjunto primero:

```bash
python3 tools/data/descargar_mnist.py
```

Y luego [GUIA_DIFUSION.md](GUIA_DIFUSION.md), que explica cómo entrenar por
tramos: los checkpoints son **sellados** —guardan todo lo que decide la
trayectoria— y reanudar da pesos **idénticos bit a bit** a no haber parado, así
que puedes cortar un entrenamiento largo en sesiones.

Un entrenamiento de difusión completo son casi **5 horas** de CPU. No lo lances
sin leer la guía antes.

---

## Adónde ir después

| Si quieres… | Ve a |
| --- | --- |
| Entender cómo encaja todo | [ARQUITECTURA.md](ARQUITECTURA.md) |
| Saber qué está hecho de verdad | [ESTADO.md](ESTADO.md) |
| Estudiar la teoría con el código delante | [TEORIA_E_IMPLEMENTACION.md](TEORIA_E_IMPLEMENTACION.md) |
| Fiarte de los números | [VERIFICACION.md](VERIFICACION.md) y [MODELOS.md](MODELOS.md) |
| Añadir una capa nueva | [ARQUITECTURA.md](ARQUITECTURA.md), sección final |
