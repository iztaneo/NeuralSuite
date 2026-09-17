# Mapa del código y orden de lectura

Este documento dice **qué hay en cada sitio** y **en qué orden leerlo** según lo
que quieras conseguir.

La regla de organización: **la interfaz y el porqué del diseño están en
`include/`; el cuerpo de cada función, en `src/`**, con las mismas subcarpetas.
Si quieres entender qué hace algo, lee la cabecera. Si quieres entender cómo lo
hace, baja al `.cpp`.

---

## El reparto

<!-- BEGIN GENERATED STATS -->
NeuralSuite son **30 459 líneas** repartidas en **148 archivos**, y **60 pruebas**.

| Carpeta | Líneas | Archivos | Contenido |
| --- | --- | --- | --- |
| `include/` | 7 128 | 52 | Interfaces, y la explicación de cada decisión |
| `src/` | 7 451 | 43 | Implementaciones |
| `tests/` | 6 659 | 1 | Las 60 pruebas, en un solo archivo |
| `apps/` | 2 878 | 8 | Los programas de entrenamiento e inferencia |
| `tools/` | 4 980 | 29 | Paridad con PyTorch y Pillow, corpus y datos |
| `demos/` | 1 037 | 12 | Una demostración por técnica |
| `benchmarks/` | 326 | 3 | Mediciones de rendimiento |
<!-- END GENERATED STATS -->

Que las pruebas ocupen casi tanto como las implementaciones no es casualidad:
es la proporción que hace posible cambiar el código sin romperlo.

---

## Tres recorridos según lo que busques

### A. «Quiero entender cómo funciona una red neuronal por dentro»

Lee en este orden. Cada paso se apoya en el anterior:

1. **`include/tensor.h`** — el tipo sobre el que gira todo: una matriz de
   `float` con forma, que puede compartir memoria con otra sin copiar.
2. **`include/parameter.h`** y **`include/module.h`** — un peso entrenable es su
   valor **y** su gradiente juntos; un módulo es un árbol de parámetros. Aquí
   está el mecanismo que impide olvidarse de un gradiente.
3. **`include/layer.h`** y **`include/layers/linear.h`** — la capa más simple,
   con su `Forward` y su `Backward`.
4. **`include/losses.h`** — cómo se mide el error y de dónde sale el primer
   gradiente.
5. **`include/optimizers.h`** — cómo ese gradiente se convierte en un cambio de
   los pesos.
6. **`demos/demo_mlp.cpp`** — las cinco piezas anteriores, juntas y funcionando.

Con eso ya sabes leer cualquier capa del proyecto.

### B. «Quiero entender el modelo de lenguaje»

1. **`include/layers/embedding.h`** — de números enteros a vectores.
2. **`include/layers/attention.h`** — la atención multi-cabeza. La cabecera
   explica el porqué; `src/layers/attention.cpp` tiene las 485 líneas del cómo,
   incluidas RoPE y la caché de claves y valores.
3. **`include/gpt.h`** — cómo se apilan los bloques y por qué el embedding de
   entrada y la salida comparten la misma matriz.
4. **`apps/train_llm.cpp`** — el bucle de entrenamiento completo, de principio a
   fin. Se lee del tirón.
5. **`apps/generate_llm.cpp`** — la generación, con la caché deslizante.

### C. «Quiero entender la generación de imágenes»

1. **`include/diffusion/schedule.h`** — qué es añadir ruido de forma controlada.
   Es el concepto sobre el que se apoya todo lo demás.
2. **`include/diffusion/resblock.h`** — el bloque que sabe en qué paso del ruido
   está.
3. **`include/diffusion/unet.h`** — cómo se baja y se sube de resolución
   guardando el detalle.
4. **`include/diffusion/sampler.h`** — cómo se pasa de ruido puro a una imagen.
5. **`apps/train_diffusion.cpp`** y **`apps/sample_diffusion.cpp`** — entrenar y
   generar.
6. **`include/latent/autoencoder.h`** y **`include/latent/gaussiana.h`** — la
   compresión sobre la que trabajará la difusión latente.

---

## Qué hay en cada carpeta

### `include/` y `src/` — la biblioteca

**Raíz: el núcleo.**

| Archivo | Qué es |
| --- | --- |
| `tensor.h` | El tipo `Tensor`: forma, memoria compartida, vistas, operaciones |
| `parameter.h` | Un peso y su gradiente en un solo objeto |
| `module.h` | Árbol de parámetros con nombres; de aquí salen guardar y cargar |
| `layer.h` | La interfaz de una capa: `Forward` y `Backward` |
| `activations.h` | ReLU, GELU, SiLU, Sigmoid, Tanh y Softmax |
| `losses.h` | Entropía cruzada y error cuadrático |
| `optimizers.h` | `SGD`, `AdamW` y la media móvil de pesos (`EMA`) |
| `autograd.h` | Motor de diferenciación automática (835 líneas) |
| `parallel.h` | Reparto de trabajo entre hilos, con resultado idéntico bit a bit |
| `serialization.h` | El formato NSF: cabecera, metadatos, tensores y suma de comprobación |
| `tokenizer.h` | Tokenizadores de caracteres y de bytes |
| `gpt.h` | El Transformer: `GPTBlock` y `GPTModel` |
| `neuralsuite.h` | Cabecera única que incluye todo, más `Sequential` |
| `artifacts.h` | Resuelve rutas bajo `release/` |

**`layers/` — las capas.** `linear`, `conv2d`, `maxpool2d`, `lstm` (con
`BiLSTM`), `attention`, `cross_attention`, `embedding`, `layernorm`, `rmsnorm`,
`groupnorm`, `swiglu`, `resample2d` (subir y bajar resolución), `residual`,
`resblock2d` y `graph_conv`.

Tres de ellas tienen una versión `*Reference` al lado —`Conv2DReference`,
`LSTMReference`, `MultiHeadAttentionReference`—: la implementación literal y
lenta que sirve para verificar que la rápida calcula lo mismo.

**`models/`** — `CRNNModel`, el modelo de OCR. El GPT vive en `gpt.h`.

**`diffusion/`** — `schedule` (el calendario de ruido), `time_embedding`,
`resblock` (bloque condicionado por el paso), `unet` y `sampler` (DDPM y DDIM).

**`latent/`** — `autoencoder` (`Codificador` y `Decodificador`) y `gaussiana`
(el latente probabilístico con su penalización KL).

**`entrenamiento/`** — `checkpoint`: el sello de la configuración, el guardado
transaccional, el estado del optimizador y la tasa de aprendizaje.

**`data/`** — `mnist` (lector del formato IDX) y `dataloader` (lotes y
particiones que comparten memoria).

**`image/`** — decodificadores propios de `png` (con su `inflate`), `jpeg`,
`bmp` y `netpbm`, más `bitmap` (escala de grises y reescalado), `binarizar`,
`enderezar` y `renglones` para el OCR.

### `apps/` — los programas

| Programa | Líneas | Qué hace |
| --- | --- | --- |
| `train_diffusion.cpp` | 697 | Entrena la difusión, con checkpoints y reanudación |
| `train_autoencoder.cpp` | 509 | Entrena el autoencoder latente |
| `train_ocr.cpp` | 392 | Entrena el OCR |
| `train_llm.cpp` | 389 | Entrena el GPT |
| `sample_diffusion.cpp` | 254 | Genera imágenes y las guarda en PNG |
| `generate_llm.cpp` | 253 | Genera texto |
| `eval_llm.cpp` | 168 | Pérdida y perplejidad sobre las tres particiones |
| `ocr_cli.cpp` | 216 | Transcribe una imagen |

### `docs/`

Las guías, la teoría y el diario de ingeniería: el índice está en
[docs/README.md](README.md). Ojo con un nombre que confunde: **`docs/referencias/`
no es bibliografía**, son las transcripciones correctas con las que
`tools/ocr/evaluar.py` mide el error del OCR. La bibliografía es
[REFERENCIAS.md](REFERENCIAS.md).

### `tests/`, `tools/`, `demos/`, `benchmarks/`

- **`tests/test_suite.cpp`**: las 60 pruebas, numeradas. Buscar «Test 47» lleva
  directo a la prueba de RoPE. Cada una explica en su comentario **qué fallo
  detecta**.
- **`tools/parity/`**: exporta desde PyTorch, calcula lo mismo en C++ y compara.
  **`tools/image/`** hace lo propio contra Pillow. **`tools/corpus/`** y
  **`tools/data/`** descargan los datos.
- **`demos/`**: doce programas cortos, uno por técnica, que se ejecutan sin
  datos externos.
- **`benchmarks/`**: mide `MatMul` por forma y el escalado con hilos,
  comprobando que el reparto no altera el resultado.

---

## Cómo se lee una capa

Todas siguen el mismo patrón, así que aprendida una, están todas:

```cpp
class Cualquiera : public Layer {
 public:
  Cualquiera(...);                      // registra sus pesos con Register()
  Tensor Forward(const Tensor& x);      // calcula y guarda lo que hará falta
  Tensor Backward(const Tensor& dout);  // devuelve dx y acumula el gradiente
};
```

Dos detalles que se repiten en todo el proyecto:

- **`Forward` guarda lo que su `Backward` necesitará.** Por eso las capas tienen
  miembros como `h1_pre_`: no son estado accidental, son la mitad del cálculo.
- **`Backward` devuelve el gradiente de la entrada y acumula el de los pesos.**
  Nunca los sobrescribe: acumular es lo que permite que una capa reciba
  gradiente por dos caminos, como en un bloque residual.

---

## Dónde está cada cosa que quizá buscas

| Busco… | Está en |
| --- | --- |
| Cómo se guarda un modelo | `include/serialization.h` y `Module::NamedParameters` |
| Por qué reanudar da el mismo resultado | `include/entrenamiento/checkpoint.h` |
| Cómo se reparte el trabajo entre hilos | `include/parallel.h` |
| Por qué la convolución es rápida | `src/layers/conv2d.cpp` (`im2col` + GEMM) |
| Qué garantiza cada prueba | `tests/test_suite.cpp` y [VERIFICACION.md](VERIFICACION.md) |
| De qué artículo sale cada pieza | [REFERENCIAS.md](REFERENCIAS.md) |
| Qué significa un término | [GLOSARIO.md](GLOSARIO.md) |
| Cómo encaja todo | [ARQUITECTURA.md](ARQUITECTURA.md) |
