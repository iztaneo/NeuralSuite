# 🧠 NeuralSuite: C++17 Deep Learning Framework from Scratch

**NeuralSuite** es un framework y suite de aprendizaje profundo escrita totalmente en **C++17 puro desde cero** (sin PyTorch, TensorFlow, BLAS, Eigen ni librerías externas de IA).

Es totalmente **multiplataforma (Linux, macOS y Windows)** y cuenta con soporte para **ejecución paralela multi-hilo en CPU (OpenMP/SIMD)**.

---

## 🛠️ Arquitectura y Capas Soportadas

1. **Perceptrón Multicapa (MLP / Redes Densas)**: Capas `Linear` ($Y = XW + b$).
2. **Redes Convolucionales (CNN)**: Capas `Conv2D` y `MaxPool2D` para visión por computador.
3. **Redes Recurrentes (LSTM)**: Capas `LSTM` para secuencias y series temporales.
4. **Transformers / LLM**: `MultiHeadAttention` (Atención Causal Multi-Cabeza), `Embedding`, `LayerNormLayer` y `GPTModel`.
5. **Optimizadores**: `AdamW` (con Weight Decay desacoplado) y `SGD` con Momentum.
6. **Funciones de Pérdida**: `CrossEntropyLoss` y `MSELoss`.

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

## 🧪 Ejecutables y Demostraciones

- `./test_suite`: Pruebas unitarias numéricas y verificación de gradientes por diferencias finitas.
- `./demo_mlp`: Entrenamiento de un clasificador MLP (XOR).
- `./demo_cnn`: Entrenamiento de una Red Convolucional (CNN 2D).
- `./demo_lstm`: Entrenamiento de una Red Recurrente (LSTM).
- `./train_llm`: Entrenamiento del LLM Transformer sobre dataset de texto.
- `./generate_llm`: Inferencia y generación de texto autorregresivo desde la consola C++.
