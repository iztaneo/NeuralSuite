# 🧠 NeuralSuite: C++17 Deep Learning Framework from Scratch

**NeuralSuite** es un framework y suite de aprendizaje profundo escrita totalmente en **C++17 puro desde cero** (sin PyTorch, TensorFlow, BLAS, Eigen ni librerías externas de IA).

Es totalmente **multiplataforma (Linux, macOS y Windows)** y cuenta con soporte para **ejecución paralela multi-hilo en CPU (OpenMP/SIMD)**.

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

**Sigue siendo software 0.x.** La cobertura de gradient checking aún no alcanza
a `Conv2D`, `LayerNorm`, `LSTM` ni `CrossEntropyLoss`; el formato de
serialización no tiene cabecera ni versión; y no hay integración continua. El
uso recomendado es educativo y de lectura del código.

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
9. **Modelo OCR**: `CRNNModel` (Conv2D + MaxPool2D + Linear) — ver la nota de
   alcance en la sección de demostraciones.

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

- `./test_suite`: Pruebas unitarias numéricas y verificación de gradientes por
  diferencias finitas (GELU, `MultiHeadAttention` completa y la matriz `wte`
  compartida por weight tying), más las invariantes de alineación entre
  parámetros y gradientes y la validación de formas e índices.

**Redes densas, convolucionales y recurrentes**

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

- `./demo_ocr`: Entrena el `CRNNModel` sobre lotes sintéticos.
- `./ocr_cli`: Ejecuta el forward del `CRNNModel` y escribe la salida a un `.txt`.

> **Alcance del OCR.** NeuralSuite **no incluye un decodificador de imagen
> PNG/JPG propio**, así que `ocr_cli` no lee los píxeles del archivo que se le
> pasa en `--image`: corre el forward sobre un tensor sintético de ruido 8×8.
> `demo_ocr` entrena igualmente sobre ruido sintético con 4 etiquetas fijas.
> Ambos sirven para ejercitar el camino `Conv2D → MaxPool2D → Linear → decode`,
> **no son reconocimiento de texto real** y su salida no debe interpretarse como
> una transcripción.

---

## 📄 Licencia

Distribuido bajo la [Licencia Apache 2.0](LICENSE).
