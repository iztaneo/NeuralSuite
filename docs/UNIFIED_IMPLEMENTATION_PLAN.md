# Plan de Implementación Unificado y Priorizado por Dificultad

Este plan unifica la hoja de ruta para la extensión de **`NeuralSuite` (C++)** y **`LLMRasec` (Python)**, cubriendo todas las familias faltantes de redes neuronales y las mejoras avanzadas de LLM, **ordenadas de menor a mayor dificultad**.

---

## 🚦 Clasificación por Dificultad y Fases de Ejecución

```text
FASE 1: Dificultad Baja 🟢 (Fácil / Implementación Rápida)
  ├── 1. Autoencoders (AE / Denoising AE)
  └── 2. Redes Residuales (ResNet Block / Skip Connections)

FASE 2: Dificultad Media 🟡 (Complejidad Intermedia)
  ├── 3. KV-Cache (Caché de Llaves y Valores para LLM)
  ├── 4. RoPE (Rotary Position Embeddings para LLM)
  ├── 5. Redes Generativas Adversarias (GANs)
  └── 6. Redes Neuronales para Grafos (GNN / GCN)

FASE 3: Dificultad Alta 🔴 (Complejidad Avanzada)
  └── 7. Modelos de Difusión (Toy DDPM)
```

---

## 📋 Detalle Técnico por Fases

### 🟢 FASE 1: Dificultad Baja (Construcción Rápida con Bloques Existentes)

#### 1. Autoencoder (`demo_autoencoder.cpp`)
- **Arquitectura**:
  - *Encoder*: `Linear(input_dim, hidden_dim)` $\to$ `ReLU` $\to$ `Linear(hidden_dim, latent_dim)`
  - *Decoder*: `Linear(latent_dim, hidden_dim)` $\to$ `ReLU` $\to$ `Linear(hidden_dim, input_dim)`
- **Función de Pérdida**: `MSELoss` (Reconstrucción $\hat{X} \approx X$).
- **Archivos a crear/modificar**:
  - `demo_autoencoder.cpp` en `NeuralSuite`.
  - Integración en `Makefile` y `CMakeLists.txt`.

#### 2. Redes Residuales - ResNet Block (`demo_resnet.cpp` & `layers/residual.h`)
- **Arquitectura**:
  - `ResidualBlock`: $y = \text{ReLU}(\text{Conv2D}_2(\text{ReLU}(\text{Conv2D}_1(x))) + x)$
- **Archivos a crear/modificar**:
  - `include/layers/residual.h` (Capa de suma shortcut $x + f(x)$).
  - `demo_resnet.cpp` en `NeuralSuite`.

---

### 🟡 FASE 2: Dificultad Media (Optimizaciones y Estructuras Compuestas)

#### 3. KV-Cache (Inferencia $O(1)$ para LLMs)
- **Objetivo**: Acelerar la generación autorregresiva de 10x a 50x.
- **Archivos a modificar**:
  - `include/layers/attention.h` (Búferes `k_cache_` y `v_cache_`).
  - `include/gpt.h` (`GPTModel::Forward` con soporte `use_cache`).
  - `generate_llm.cpp` en C++ y `generate.py` en Python.

#### 4. RoPE - Rotary Position Embeddings
- **Objetivo**: Reemplazar la tabla estática `wpe_` por rotaciones matriciales en el plano complejo sobre $Q$ y $K$.
- **Archivos a modificar**:
  - `include/layers/attention.h` (Función `ApplyRoPE`).
  - `src/model.py` en Python (`apply_rotary_emb`).

#### 5. GAN - Redes Generativas Adversarias (`demo_gan.cpp`)
- **Arquitectura**:
  - *Generador $G(z)$*: `Sequential` (Ruido $z \to$ Datos sintéticos).
  - *Discriminador $D(x)$*: `Sequential` (Clasificación Real/Falso $\to$ Sigmoid).
- **Entrenamiento Minimax**: Actualización alternada de $D$ y $G$.

#### 6. GNN - Redes para Grafos (`demo_gnn.cpp` & `layers/graph_conv.h`)
- **Arquitectura**:
  - `GraphConv`: $H^{(l+1)} = \text{ReLU}(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W)$
- **Entrenamiento**: Clasificación de nodos sobre matrices de adyacencia de grafos.

---

### 🔴 FASE 3: Dificultad Alta (Síntesis Probabilística Avanzada)

#### 7. Modelo de Difusión (`demo_diffusion.cpp`)
- **Arquitectura**:
  - Proceso estocástico de adición de ruido gaussiano (*Forward Process*).
  - Red U-Net simplificada para predecir y eliminar el ruido (*Reverse Denoising*).

---

## 🧪 Plan de Verificación y Entregables

1. Cada fase incluirá su propio programa demostrativo ejecutable (`demo_autoencoder`, `demo_resnet`, `demo_gan`, `demo_gnn`, `demo_diffusion`).
2. Todos los binarios se vincularán automáticamente a `libneuralsuite.a` y `libneuralsuite.so`.
3. Se verificará que las pérdidas (*Loss*) desciendan limpiamente a cero en cada demo.
