# Plan de Implementación Futuro: KV-Cache y RoPE (Rotary Position Embeddings)

Este plan documenta la hoja de ruta para incorporar dos de las técnicas más importantes de los LLMs modernos (**RoPE** y **KV-Cache**) tanto en nuestra suite de C++ (`NeuralSuite`) como en Python (`LLMRasec`).

## 🎯 Objetivos de la Mejora
1. **KV-Cache (Inferencia Ultrarrápida $O(1)$)**:
   - Al generar texto token por token, evitar recalcular la atención de todo el contexto pasado.
   - Reutilizar los tensores $K$ y $V$ calculados previamente en memoria.
   - **Resultado**: Incremento de velocidad de generación autorregresiva de 10x a 50x.

2. **RoPE (Rotary Position Embedding - Estilo LLaMA / Mistral / Gemma)**:
   - Reemplazar la tabla posicional estática `wpe_` por rotaciones complejas sobre los tensores $Q$ y $K$.
   - **Resultado**: Permitir extrapolación de la ventana de contexto a secuencias más largas sin degradación de atención.

---

## 🛠️ Cambios Planeados

### Componente 1: Suite C++ (`NeuralSuite`)

#### [attention.h](file:///home/rasec/Documents/Proyectos/NeuralSuite/include/layers/attention.h)
- **RoPE**: Añadir función `ApplyRoPE(Tensor& q, Tensor& k, int seq_offset)` que aplica la matriz de rotación 2D sobre pares de canales de cada cabeza.
- **KV-Cache**: Agregar tensores `k_cache_` y `v_cache_` dentro de `MultiHeadAttention` para almacenar los estados pasados durante la inferencia (`is_generation = true`).

#### [gpt.h](file:///home/rasec/Documents/Proyectos/NeuralSuite/include/gpt.h)
- Actualizar `GPTModel::Forward` para aceptar una bandera opcional `use_cache` y devolver el token predicho reutilizando la caché posicional sin recalcular el pasado.

#### [generate_llm.cpp](file:///home/rasec/Documents/Proyectos/NeuralSuite/generate_llm.cpp)
- Conectar el bucle de generación para usar `use_cache = true`, acelerando la salida de texto en tiempo real.

---

### Componente 2: Proyecto Python (`LLMRasec`)

#### [model.py](file:///home/rasec/Documents/Proyectos/LLM-rasec/src/model.py)
- Implementar la función `apply_rotary_emb(q, k, freqs_cis)` y la gestión del búfer `past_key_values` en `CausalSelfAttention` y `GPT`.

#### [generate.py](file:///home/rasec/Documents/Proyectos/LLM-rasec/generate.py)
- Actualizar la llamada de generación para reutilizar la caché de PyTorch.
