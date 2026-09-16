# Guía: entrenar y usar el modelo de lenguaje

NeuralSuite incluye un Transformer tipo GPT que se entrena, evalúa y genera texto
sin Python ni bibliotecas externas. Esta guía recorre el flujo completo con el
modelo de referencia en español.

Todos los comandos se lanzan desde la raíz del repositorio, después de `make`.

---

## 1. Preparar el corpus

```bash
python3 tools/corpus/preparar_corpus.py
```

Descarga unos 7 MB de obras de dominio público de Project Gutenberg y produce
tres particiones en `corpus/es/`:

| Archivo | Contenido |
| --- | --- |
| `train.txt` | Cervantes, Clarín, Pardo Bazán, Unamuno |
| `val.txt` | Los mismos autores, una porción interior de cada libro |
| `test.txt` | **Blasco Ibáñez**, un autor que nunca aparece en entrenamiento |

El conjunto de prueba es un autor entero apartado a propósito: así se mide si el
modelo generaliza al español, y no solo si memoriza a sus autores. Detalles en
[tools/corpus/README.md](../tools/corpus/README.md).

Es el único paso que usa Python. Entrenar y generar no lo necesitan.

---

## 2. Entrenar

El modelo de referencia (858 K parámetros, 4 capas, 4 cabezas, embedding de 128,
contexto de 128 caracteres):

```bash
./bin/train_llm --data_path corpus/es/train.txt --out_file release/es_base.bin --vocab_file release/es_base_vocab.txt --max_iters 5000 --block_size 128 --n_embd 128 --n_layer 4 --n_head 4 --batch_size 16
```

Tarda unos 22 minutos en un Apple M5. Para ver la pérdida de validación mientras
entrena, añade `--val_path corpus/es/val.txt --eval_cada 250`.

| Opción | Por defecto | Qué hace |
| --- | --- | --- |
| `--data_path` | `sample_data/input.txt` | Texto de entrenamiento |
| `--max_iters` | 1000 | Iteraciones |
| `--batch_size` | 16 | Tamaño del lote |
| `--block_size` | 64 | Longitud del contexto |
| `--n_layer`, `--n_head`, `--n_embd` | 4, 4, 64 | Tamaño del modelo |
| `--learning_rate` | 0.003 | Tasa inicial, con decaimiento coseno |
| `--rope` | desactivado | Posición por rotación (RoPE) en vez de aprendida |
| `--val_path`, `--eval_cada` | —, 250 | Evaluar validación durante el entrenamiento |
| `--out_file`, `--vocab_file` | `release/model_cpp.bin`, `release/vocab_cpp.txt` | Dónde guardar |

El tokenizador trabaja **a nivel de carácter**, y el vocabulario se guarda junto
al modelo: cada modelo tiene que usarse con su propio vocabulario.

---

## 3. Evaluar

```bash
./bin/eval_llm --model release/es_base.bin --vocab release/es_base_vocab.txt --corpus corpus/es --block_size 128 --n_embd 128 --completo
```

Mide pérdida y perplejidad en las tres particiones. Con `--completo` recorre las
particiones enteras en vez de una muestra. Los resultados del modelo de
referencia:

| Partición | Perplejidad |
| --- | --- |
| train | 5.69 |
| val | 5.84 |
| **test** (Blasco Ibáñez, nunca visto) | **6.23** |

El orden `train < val < test` es el que debe ser. Un modelo que memorizara
tendría un salto mucho mayor en el autor nunca visto.

---

## 4. Generar texto

```bash
./bin/generate_llm --model_path release/es_base.bin --vocab_path release/es_base_vocab.txt --block_size 128 --n_embd 128 --prompt "En un lugar de la Mancha" --max_new_tokens 300 --temperature 0.8
```

| Opción | Por defecto | Qué hace |
| --- | --- | --- |
| `--prompt` | `First Citizen:` | Texto inicial |
| `--max_new_tokens` | 200 | Caracteres a generar |
| `--temperature` | 0.7 | Más alta, más variado; más baja, más conservador |
| `--rope` | desactivado | **Tiene que coincidir** con cómo se entrenó el modelo |
| `--no_cache` | desactivado | Desactiva el KV-Cache (útil solo para comparar) |

**Las opciones de arquitectura** (`--block_size`, `--n_layer`, `--n_head`,
`--n_embd`, `--rope`) **tienen que coincidir con las del entrenamiento.** Los
pesos se guardan en formato NSF con la arquitectura en los metadatos, así que un
desajuste se rechaza con un mensaje, en vez de generar basura.

La generación usa **KV-Cache con ventana deslizante**: cuando el texto supera el
contexto, la caché descarta lo más antiguo sin reconstruirse. Medido: 0.06 ms por
token.

---

## Qué esperar

El modelo de referencia genera **español reconocible**: concordancia de
artículos, terminaciones verbales, acentos e incluso la raya de diálogo de la
novela española. Pero con palabras que no existen, porque a nivel de carácter y
con 858 K parámetros aprende cómo **suena** el español, no lo que **significa**.

Dos límites actuales:

- **Tokenizador de caracteres.** Un tokenizador BPE acortaría las secuencias; está
  pendiente en la Fase 08.
- **`train_llm` no reanuda.** Los entrenadores de difusión y del autoencoder
  tienen checkpoints sellados y reanudación exacta; el del LLM todavía no los
  usa.
