# KV-Cache y RoPE

Este documento cubría dos técnicas planteadas juntas. **Ya no están en el mismo
estado**, así que conviene leerlas por separado:

| | Estado |
| --- | --- |
| **KV-Cache** | **Hecho y medido.** Queda una mejora de almacenamiento. |
| **RoPE** | **Pendiente de verdad.** Se retiró un esbozo que no hacía nada. |

---

## 1. KV-Cache — hecho

Al generar texto token por token, la atención no recalcula el contexto pasado:
reutiliza los tensores K y V ya calculados.

Está implementado en `MultiHeadAttention` (`k_cache_`, `v_cache_`), expuesto por
`GPTModel::ForwardWithKVCache(token, pos)` y `ClearKVCache()`, y conectado en
[apps/generate_llm.cpp](../apps/generate_llm.cpp), que además acepta
`--no_cache` para desactivarlo.

**Medido** con 4 capas, 4 cabezas, `n_embd` 128, generando 64 tokens tras un
prompt de 16:

| | Total | Por token |
| --- | --- | --- |
| Con KV-Cache | 23.3 ms | 0.36 ms |
| Sin KV-Cache | 224.2 ms | 3.50 ms |
| | | **9.6× más rápido** |

Esa cifra es **dentro de la ventana**. Dos correcciones importantes que
aparecieron al comprobarlo a fondo:

- La afirmación anterior —«las dos rutas generan exactamente la misma
  secuencia»— era una comprobación **demasiado débil**: el argmax absorbe
  diferencias pequeñas. Comparando *logits* apareció un defecto real:
  `generate_llm` reinyectaba el último token del prompt y lo metía dos veces en
  la caché, con 0.056 de diferencia frente a la ruta de referencia. Ya está
  arreglado y el test 38 compara logits paso a paso.
- **Pasada la ventana, la caché deja de acelerar**: 1.29 ms/token frente a 0.13
  dentro, con 73 reconstrucciones en 100 tokens. Con posiciones aprendidas y
  absolutas, deslizar cambia la posición de cada token y las K y V guardadas
  dejan de valer. Es el argumento más fuerte a favor de RoPE.

### Lo que queda del KV-Cache

- [ ] **Almacenamiento contiguo.** Hoy es `std::vector<std::vector<float>>`: un
      vector por posición, cada uno con su propia reserva. Un único bloque
      contiguo evitaría los saltos de memoria y las reservas por token. Es la
      única tarea abierta de esta parte, y está en el roadmap como *KV cache
      contiguo* (Fase 09).

---

## 2. RoPE — pendiente

**Rotary Position Embedding**: en vez de sumar una tabla posicional aprendida,
se rotan los pares de canales de Q y K en función de la posición. Es lo que usan
LLaMA, Mistral y Gemma. La ventaja no es velocidad sino **extrapolación**: el
modelo tolera secuencias más largas que las vistas en entrenamiento, porque la
posición entra como una relación entre pares y no como una fila de una tabla que
nunca se entrenó.

### Por qué no está

Existió aquí un `ApplyRoPE()` que **ningún forward llamaba**. Se retiró en vez de
dejarlo: una función así sugiere una capacidad que el modelo no tiene, y es peor
que su ausencia porque nadie sabe que no hace nada. La posición sigue llegando
por embeddings aprendidos (`wpe_`).

### Lo que exige implementarlo

- [ ] Rotar Q y K por posición dentro de `MultiHeadAttention`, **antes** del
      producto de atención y **después** de la proyección `c_attn_`.
- [ ] Propagar por esa rotación en el backward, con su comprobación por
      diferencias finitas. La rotación es lineal, así que su gradiente es la
      rotación inversa; no es difícil, pero sin gradient check no vale.
- [ ] Decidir qué pasa con `wpe_`. RoPE la hace redundante: mantener las dos es
      sumar dos señales de posición distintas. Retirarla cambia el número de
      parámetros y **rompe la compatibilidad de los pesos guardados**, así que
      hay que versionar el formato `.nsf` o convertir.
- [ ] Encajarlo con el KV-Cache. La rotación depende de la posición absoluta, de
      modo que `ForwardWithKVCache(token, pos)` debe rotar con el `pos` real y no
      con el índice dentro de la caché. Es el error clásico de esta combinación.
- [ ] Actualizar la referencia en PyTorch (`LLMRasec`), o la paridad deja de ser
      válida: si el C++ rota y el oráculo no, la comparación falla por diseño y
      no por defecto.

### Nota sobre el orden

RoPE **no** debería hacerse antes que el BPE. Ambos tocan la longitud de
secuencia, pero el BPE la reduce 3–4× de entrada, mientras que RoPE sólo ayuda a
extrapolar más allá de lo entrenado — un problema que aún no tenemos.

---

## Historial

La versión anterior de este documento presentaba las dos técnicas como trabajo
futuro y apuntaba a rutas de otra máquina (`/home/rasec/...`) y a un
`generate_llm.cpp` en la raíz que hoy vive en `apps/`. El KV-Cache llevaba
tiempo hecho sin que el documento lo dijera, y RoPE no aparecía como pendiente
abierto en el roadmap, sólo como una decisión ya tomada — que es justo lo que lo
hacía invisible.
