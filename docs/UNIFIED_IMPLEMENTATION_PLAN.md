# Plan de Implementación Unificado: Paridad Total en Ambos Proyectos

> **Estado: 6 de 7 entregables hechos en C++. Sólo falta RoPE.**
>
> | | Entregable | C++ |
> | --- | --- | --- |
> | 1 | Autoencoders | ✅ `demos/demo_autoencoder.cpp` |
> | 2 | Redes residuales | ✅ `include/layers/residual.h` |
> | 3 | KV-Cache | ✅ medido: 9.6× más rápido, misma secuencia |
> | 4 | **RoPE** | ❌ **lo único pendiente** |
> | 5 | GAN | ✅ `demos/demo_gan.cpp` |
> | 6 | GNN / GCN | ✅ `include/layers/graph_conv.h` |
> | 7 | Difusión (toy DDPM) | ✅ `demos/demo_diffusion.cpp` |
>
> Lo que exige RoPE está detallado en
> [FUTURE_PLAN_KVCACHE_ROPE.md](FUTURE_PLAN_KVCACHE_ROPE.md), y ahora también
> figura como pendiente abierto en [ROADMAP.md](ROADMAP.md), que es donde se
> mira. Este documento describe el plan original y se conserva por eso; el
> estado vigente está arriba.
>
> Nota: `generate_llm.cpp` se menciona más abajo en la raíz del repositorio;
> hoy vive en `apps/`.

Este plan establece la hoja de ruta para implementar cada una de las arquitecturas y optimizaciones **en AMBOS PROYECTOS en paralelo**:
- **`NeuralSuite`** (C++17 puro desde cero).
- **`LLMRasec`** (Python / PyTorch).

---

## 🚦 Clasificación por Dificultad y Fases de Ejecución

```text
FASE 1: Dificultad Baja 🟢 (Fácil / Implementación Rápida)
  ├── 1. Autoencoders (AE / Denoising AE) [C++ & Python]
  └── 2. Redes Residuales (ResNet Block / Skip Connections) [C++ & Python]

FASE 2: Dificultad Media 🟡 (Complejidad Intermedia)
  ├── 3. KV-Cache (Caché de Llaves y Valores para LLM) [C++ & Python]
  ├── 4. RoPE (Rotary Position Embeddings para LLM) [C++ & Python]
  ├── 5. Redes Generativas Adversarias (GANs) [C++ & Python]
  └── 6. Redes Neuronales para Grafos (GNN / GCN) [C++ & Python]

FASE 3: Dificultad Alta 🔴 (Complejidad Avanzada)
  └── 7. Modelos de Difusión (Toy DDPM) [C++ & Python]
```

---

## 📋 Detalle de Entregables por Proyecto y Fase

### 🟢 FASE 1: Dificultad Baja

#### 1. Autoencoders (AE / Denoising AE)
- **C++ (`NeuralSuite`)**:
  - Crear `demo_autoencoder.cpp` compilado contra `libneuralsuite.so`.
  - Probar compresión y reconstrucción de datos con capas `Linear` y `MSELoss`.
- **Python (`LLMRasec`)**:
  - Crear `src/autoencoder.py` y `demo_autoencoder.py` en PyTorch.
  - Verificar convergencia de pérdida de reconstrucción idéntica.

#### 2. Redes Residuales (ResNet Block)
- **C++ (`NeuralSuite`)**:
  - Crear `include/layers/residual.h` (Capa de suma shortcut $y = \text{ReLU}(f(x) + x)$).
  - Crear `demo_resnet.cpp` y agregar al `Makefile`.
- **Python (`LLMRasec`)**:
  - Crear `src/resnet.py` y `demo_resnet.py` en PyTorch.

---

### 🟡 FASE 2: Dificultad Media

#### 3. KV-Cache (Inferencia $O(1)$ para LLMs)
- **C++ (`NeuralSuite`)**:
  - Modificar `include/layers/attention.h` y `include/gpt.h` para almacenar tensores `k_cache_` y `v_cache_`.
  - Actualizar `generate_llm.cpp`.
- **Python (`LLMRasec`)**:
  - Modificar `src/model.py` (`CausalSelfAttention`) y `generate.py` para usar caché de llaves/valores.

#### 4. RoPE (Rotary Position Embeddings)
- **C++ (`NeuralSuite`)**:
  - Añadir rotaciones vectoriales complejas en $Q$ y $K$ en `MultiHeadAttention::Forward`.
- **Python (`LLMRasec`)**:
  - Añadir `apply_rotary_emb` en `src/model.py`.

#### 5. Redes Generativas Adversarias (GANs)
- **C++ (`NeuralSuite`)**:
  - Crear `demo_gan.cpp` con redes *Generador* y *Discriminador* enfrentadas.
- **Python (`LLMRasec`)**:
  - Crear `src/gan.py` y `demo_gan.py` en PyTorch.

#### 6. Redes Neuronales para Grafos (GNN / GCN)
- **C++ (`NeuralSuite`)**:
  - Crear `include/layers/graph_conv.h` y `demo_gnn.cpp`.
- **Python (`LLMRasec`)**:
  - Crear `src/gnn.py` y `demo_gnn.py`.

---

### 🔴 FASE 3: Dificultad Alta

#### 7. Modelos de Difusión (Toy DDPM)
- **C++ (`NeuralSuite`)**:
  - Crear `demo_diffusion.cpp` (Proceso de adición y eliminación de ruido gaussiano).
- **Python (`LLMRasec`)**:
  - Crear `src/diffusion.py` y `demo_diffusion.py`.

---

## 🧪 Plan de Verificación

En cada paso de la implementación:
1. Se ejecutarán las demos en **C++** y en **Python**.
2. Se verificará que ambas salidas, pérdidas (*Loss*) y comportamientos de convergencia sean matemáticamente equivalentes.
3. Se confirmará y publicará el código en los repositorios de GitHub correspondientes (`NeuralSuite` y `LLMRasec`).
