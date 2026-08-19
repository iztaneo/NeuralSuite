# 🧠 NeuralSuite: C++17 Deep Learning Framework from Scratch

[![CI](https://github.com/iztaneo/NeuralSuite/actions/workflows/ci.yml/badge.svg)](https://github.com/iztaneo/NeuralSuite/actions/workflows/ci.yml)

**NeuralSuite** es un framework y suite de aprendizaje profundo escrita totalmente en **C++17 puro desde cero** (sin PyTorch, TensorFlow, BLAS, Eigen ni librerías externas de IA).

Es totalmente **multiplataforma (Linux, macOS y Windows)** y paraleliza en CPU con `std::thread`, sin depender de OpenMP.

---

## ⚠️ Estado del proyecto: fase experimental 0.x

Los dos defectos que invalidaban el entrenamiento del Transformer **están
corregidos y cubiertos por pruebas de regresión** (`./test_suite`):

- `MultiHeadAttention` no sobrescribía `GetGradients()`, así que heredaba una
  lista vacía: cada `GPTBlock` exponía 12 parámetros frente a 8 gradientes y el
  optimizador aplicaba el gradiente de una capa a los pesos de otra, leyendo
  fuera de rango al agotarse la lista más corta.
- El *weight tying* del `GPTModel` perdía la contribución de la cabeza de
  salida, porque `Embedding::Backward()` reinicia el acumulador de gradiente
  después de que el modelo ya la hubiera sumado.

Los optimizadores ahora rechazan listas de parámetros y gradientes que no
casen, en lugar de continuar en silencio, y `Tensor`, `MatMul`, `Embedding` y
`MultiHeadAttention` validan formas e índices.

La capa `LSTM` **se reimplementó por completo**. La versión anterior no era una
LSTM: su `Forward()` no usaba ninguno de los cuatro parámetros, no tenía
puertas, ignoraba el estado de celda y solo leía el primer elemento del vector
de entrada; su `Backward()` devolvía siempre ceros, de modo que la capa no
aprendía y además cortaba la propagación hacia capas anteriores. Ahora
implementa la celda estándar (puertas `i`, `f`, `g`, `o` y estado `c`) con
retropropagación a través del tiempo, verificada por diferencias finitas sobre
los cuatro tensores de parámetros y sobre el gradiente de entrada.

Las operaciones diferenciables del núcleo tienen gradient check por diferencias
finitas: `GELU`, `MultiHeadAttention`, la matriz `wte` compartida por weight
tying, `LSTM` (parámetros y gradiente de entrada), `Conv2D` (con stride y
padding activos), `LayerNorm` (respecto de `x`, `gamma` y `beta`),
`CrossEntropyLoss`, `MSELoss`, `MaxPool2D`, `ResidualBlock` y `GraphConv`. Cada
comprobación se validó introduciendo defectos deliberados para confirmar que
efectivamente los detecta.

Además, el `GPTModel` y la capa `LSTM` se comparan contra la implementación de
referencia en PyTorch ([LLMRasec](https://github.com/iztaneo/LLMRasec)) con los
mismos pesos y las mismas entradas: coinciden en pérdida, salidas y gradientes
dentro del redondeo de float32. Ver [`tools/parity/`](tools/parity/README.md).
Esta comprobación detecta errores que el gradient checking no puede ver, porque
éste solo confirma que el `Backward` deriva el `Forward` escrito, no que ese
`Forward` sea lo que dice ser.

`Tensor` usa almacenamiento compartido, de modo que `View()` reinterpreta los
ejes sin copiar. Aplanar `[B, T, C]` a `[B*T, C]` era hasta ahora una reserva
más un `memcpy` completo en cada capa densa; ahora es gratis. Medido sobre un
paso de entrenamiento del GPT (B=8, T=64, 4 capas, n_embd=128): la memoria
reservada baja de 118 MB a 72,6 MB. El tiempo por paso no cambia de forma
apreciable — el cuello de botella es el cómputo, no las reservas.

Los pesos entrenables viven en un `Parameter`, que reúne el valor y su
gradiente en un solo objeto, y las capas los declaran una vez con `Register()`.
Antes cada capa exponía dos listas paralelas que el optimizador recorría por
índice, y que una capa declarase sus pesos y olvidase sus gradientes es
exactamente el defecto que invalidaba el entrenamiento del Transformer. Ahora
ambas listas se derivan de la misma declaración.

Existe un **motor de diferenciación automática** (`include/autograd.h`) con doce
primitivas verificadas por diferencias finitas. Deduce los gradientes del paso
hacia delante en vez de que cada capa escriba su `Backward()` a mano, que es
donde aparecieron los dos defectos que invalidaban el entrenamiento.
`./demo_autograd` entrena XOR sin una sola derivada escrita, y `LayerNorm` está
compuesta de primitivas: su gradiente no lo escribió nadie y coincide con la
implementación manual. Las capas
existentes **todavía no están migradas**: el motor debía estar comprobado antes
de reescribir sobre él lo que ya funciona.

`MatMul` es el 80% del tiempo de un paso de entrenamiento —medido, no supuesto—
y reparte sus filas entre hilos. Cada hilo escribe filas que nadie más toca, así
que no hay reducción y **el resultado es idéntico bit a bit** al de un solo
hilo: la comparación contra PyTorch devuelve exactamente los mismos números.
Sobre un Apple M5, un paso de entrenamiento pasa de 119 ms a unos 27 ms —unas
4.5 veces— combinando reparto dinámico entre hilos y un kernel que calcula
cuatro filas a la vez. Las cifras salen de
`./benchmark`, que además comprueba en cada ejecución que el reparto no altera
el resultado.

Cada push ejecuta en [integración continua](.github/workflows/ci.yml) la
compilación en Linux (GCC y Clang), macOS y Windows, en modo `Debug` y
`Release`; la suite numérica; las demos; un entrenamiento del LLM de extremo a
extremo; y la comparación contra PyTorch.

El tokenizador reserva el índice 0 para `<UNK>`. Antes un símbolo fuera del
vocabulario se codificaba como token 0, que era un carácter válido: con el
vocabulario `{a, b, c}`, `"axc"` volvía como `"aac"`. Existe además un
`ByteTokenizer` de 256 símbolos que, por construcción, no puede encontrarse un
símbolo desconocido y reproduce cualquier UTF-8 —japonés o emoji incluidos— sin
reentrenarse.

Los pesos se guardan en formato **NSF**, con número mágico, versión, los
metadatos de la arquitectura, cada tensor con su nombre y forma, y una suma de
comprobación. Antes se volcaban floats crudos sin cabecera y cargar un
checkpoint de otra arquitectura no daba ningún error: el modelo se quedaba con
datos sin sentido. Ahora la carga rechaza —diciendo por qué— un archivo de otra
arquitectura, truncado, alterado o sin cabecera.

**Sigue siendo software 0.x.** Cada capa implementa su `Backward()` a mano, que
es donde aparecieron los defectos que costó más encontrar. El uso recomendado es
educativo y de lectura del código.

El estado detallado de cada fase, con los defectos encontrados y lo que queda
pendiente, está en [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## 🛠️ Arquitectura y Capas Soportadas

**Núcleo**

1. **Perceptrón Multicapa (MLP / Redes Densas)**: Capas `Linear` ($Y = XW + b$).
2. **Redes Convolucionales (CNN)**: Capas `Conv2D` y `MaxPool2D` para visión por computador.
3. **Redes Recurrentes (LSTM)**: Capas `LSTM` para secuencias y series temporales.
4. **Transformers / LLM**: `MultiHeadAttention` (Atención Causal Multi-Cabeza), `Embedding`, `LayerNormLayer` y `GPTModel`.
5. **Optimizadores**: `AdamW` (con Weight Decay desacoplado) y `SGD` con Momentum.
6. **Funciones de Pérdida**: `CrossEntropyLoss` y `MSELoss`.

**Capas y modelos adicionales**

7. **Bloques residuales**: `Residual` (base de la demo ResNet).
8. **Redes de grafos**: `GraphConv` (base de la demo GNN).
9. **Modelo OCR**: `CRNNModel` (Conv×3 → `BiLSTM` → `Linear`), que lee una línea
   de texto y devuelve una predicción por columna — ver la nota de alcance en la
   sección de demostraciones.

---

## 📖 Documentación Teórica y Matemática

El repositorio incluye guías técnicas autónomas y exhaustivas:

- [`DOCS_MATHEMATICS.md`](DOCS_MATHEMATICS.md): Derivación matemática completa del paper *Attention Is All You Need* (Vaswani et al., 2017) y Transformers Decoder-Only (GPT).
- [`DOCS_NEURAL_NETWORKS.md`](DOCS_NEURAL_NETWORKS.md): Guía teórica sobre MLPs, CNNs, LSTMs, pérdidas y optimizadores.
- [`DOCS_PROGRAMMING_CPP.md`](DOCS_PROGRAMMING_CPP.md): Patrones de diseño C++17, gestión de memoria RAII y OpenMP.

---

## 🚀 Compilación y Uso Multiplataforma

### En Linux y macOS (Usando `Makefile`)
```bash
make
```

### En Windows, Linux y macOS (Usando `CMake`)
```bash
cmake -B build
cmake --build build
```

El build por defecto es `Release` y es portable: no usa `-march=native`, así que
el binario resultante funciona en otras máquinas. Para desarrollo conviene el
build con comprobaciones, que activa la validación de índices de
`Tensor::operator[]`:

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
```

Opciones disponibles:

| Opción                      | Por defecto | Efecto                                                 |
| --------------------------- | ----------- | ------------------------------------------------------ |
| `NEURALSUITE_NATIVE_ARCH`   | `OFF`       | `-march=native`: más rápido en esta CPU, no portable    |
| `NEURALSUITE_FAST_MATH`     | `OFF`       | `-ffast-math`: relaja IEEE-754, estorba al verificar    |

El paralelismo lo aporta `include/parallel.h` con la biblioteca estándar, así
que no hace falta OpenMP: antes los `pragma omp` se ignoraban en macOS con
AppleClang y todo corría en un solo hilo. `parallel::ThreadCount()` permite
fijar el número de hilos, y ponerlo a 1 desactiva el reparto. Para medir
rendimiento en local:

```bash
cmake -B build-bench -DNEURALSUITE_NATIVE_ARCH=ON -DNEURALSUITE_FAST_MATH=ON
```

---

## 📦 Artefactos de entrenamiento: `release/`

Todo modelo entrenado (`*.ns`, `*.bin`) y todo vocabulario generado se escribe
bajo [`release/`](release/README.md), **nunca en la raíz del repositorio**. El
directorio se crea solo y su contenido está excluido del control de versiones:
los checkpoints que se quieran distribuir se adjuntan a un
[GitHub Release](https://github.com/iztaneo/NeuralSuite/releases), que es el
mecanismo pensado para binarios y no infla el historial de git.

En código la ruta se resuelve con `ReleasePath()` de `include/artifacts.h`:

```cpp
const std::string path = ReleasePath("mi_modelo.ns");  // -> "release/mi_modelo.ns"
if (!model.Save(path)) { /* el fallo se reporta, no se ignora */ }
```

Las rutas son relativas al directorio de trabajo, así que conviene lanzar los
ejecutables desde la raíz del repositorio.

---

## 🧪 Ejecutables y Demostraciones

Todos los ejecutables están registrados en `CMakeLists.txt`, que es la única
fuente de verdad del build: ninguna demo debe compilarse a mano.

**Pruebas**

- `./test_suite`: 19 pruebas numéricas — gradient checks por diferencias finitas
  de las operaciones diferenciables (ver la sección de estado), invariantes de
  alineación entre parámetros y gradientes, semántica de `Tensor`, y validación
  de formas e índices. Devuelve un código de salida distinto de cero si alguna
  falla.

**Redes densas, convolucionales y recurrentes**

- `./demo_autograd`: entrenamiento de XOR con diferenciación automática,
  sin derivadas escritas a mano.
- `./demo_adaline`: Regla de aprendizaje Adaline / Widrow-Hoff.
- `./demo_mlp`: Entrenamiento de un clasificador MLP (XOR).
- `./demo_cnn`: Entrenamiento de una Red Convolucional (CNN 2D).
- `./demo_lstm`: Entrenamiento de una Red Recurrente (LSTM).
- `./demo_sequential`: Uso del contenedor `Sequential` para apilar capas.
- `./demo_resnet`: Bloques residuales.
- `./demo_gnn`: Convolución sobre grafos.

**Modelos generativos**

- `./demo_autoencoder`: Autoencoder encoder/decoder.
- `./demo_gan`: Generador y discriminador adversarios.
- `./demo_diffusion`: Denoiser de difusión.

**LLM / Transformer**

- `./train_llm`: Entrenamiento del LLM Transformer sobre dataset de texto.
- `./generate_llm`: Inferencia y generación de texto autorregresivo desde la consola C++.

**OCR**

- `./demo_ocr`: Entrena el `CRNNModel` a transcribir líneas sintéticas y
  comprueba, al final, cuántas lee exactamente.
- `./ocr_cli`: Ejecuta el forward del `CRNNModel` y escribe la salida a un `.txt`.

> **Alcance del OCR.** NeuralSuite **no incluye un decodificador de imagen
> PNG/JPG propio**, así que `ocr_cli` no lee los píxeles del archivo que se le
> pasa en `--image`: corre el forward sobre una línea generada por
> `SynthTextGenerator`. Los glifos de esas líneas son patrones binarios
> derivados del índice de cada clase, **no una tipografía real**, de modo que
> nada de esto es reconocimiento de texto del mundo real y su salida no debe
> interpretarse como una transcripción.
>
> Lo que sí es real es el problema y la arquitectura: la red recibe la imagen de
> una línea entera y devuelve la secuencia de caracteres, una predicción por
> cada cuatro columnas, con la misma estructura que la implementación de
> referencia en PyTorch. `demo_ocr` transcribe las 4 líneas del lote sin errores
> tras 120 épocas.

---

## 📄 Licencia

Distribuido bajo la [Licencia Apache 2.0](LICENSE).
