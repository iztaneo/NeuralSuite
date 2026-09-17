# Estado por componente

Qué existe, qué está verificado, qué está **integrado en un modelo** y qué ha
**entrenado de verdad**. Son cuatro cosas distintas, y confundirlas es lo que
hizo que un documento anunciara «RoPE pendiente» meses después de terminarlo.

Las columnas significan:

| Columna | Qué quiere decir |
| --- | --- |
| **Implementado** | El código existe y compila |
| **Verificado** | Diferencias finitas, y **P** si además hay paridad contra PyTorch |
| **Integrado** | Lo usa algún modelo o aplicación, no solo una prueba |
| **Entrenado** | Ha participado en un entrenamiento real cuyos resultados están medidos |

Estado medido en el commit de este documento. La forma de comprobarlo está en
[VERIFICACION.md](VERIFICACION.md); lo que falta, en [ROADMAP.md](ROADMAP.md).

---

## Núcleo

| Componente | Implementado | Verificado | Integrado | Entrenado |
| --- | :---: | :---: | :---: | :---: |
| `Tensor` con vistas sin copia | ✅ | ✅ | ✅ | ✅ |
| `Parameter` y `Module` | ✅ | ✅ | ✅ | ✅ |
| Motor de autograd (12 primitivas) | ✅ | ✅ | ⚠️ solo demos | — |
| `parallel.h` (idéntico bit a bit) | ✅ | ✅ | ✅ | ✅ |
| Serialización NSF | ✅ | ✅ | ✅ | ✅ |
| Checkpoint sellado y transaccional | ✅ | ✅ | ✅ | ✅ |
| `DataLoader` y lector de MNIST | ✅ | ✅ | ✅ | ✅ |
| Tokenizador de caracteres y de bytes | ✅ | ✅ | ✅ | ✅ |

**El autograd está verificado pero no lo usa ningún modelo.** Fue una decisión
medida, no un olvido: las parejas `LinearAutograd` y `EmbeddingAutograd`
resultaron entre 2.7× y 20× más lentas y **encontraron cero defectos**. El
detalle, en [AUTOGRAD_CAPAS.md](AUTOGRAD_CAPAS.md).

---

## Capas

| Componente | Implementado | Verificado | Integrado | Entrenado |
| --- | :---: | :---: | :---: | :---: |
| `Linear` | ✅ | ✅ **P** | ✅ | ✅ |
| `Conv2D` (`im2col`) | ✅ | ✅ **P** | ✅ | ✅ |
| `MaxPool2D` | ✅ | ✅ | ✅ OCR | ✅ |
| `LSTM` y `BiLSTM` | ✅ | ✅ **P** | ✅ OCR | ✅ |
| `Embedding` | ✅ | ✅ **P** | ✅ GPT | ✅ |
| `MultiHeadAttention` | ✅ | ✅ **P** | ✅ GPT | ✅ |
| RoPE | ✅ | ✅ **P** | ✅ GPT, **opcional** (`--rope`) | ⚠️ el modelo de referencia usa posición aprendida |
| KV-Cache con ventana deslizante | ✅ | ✅ | ✅ `generate_llm` | n/a |
| `LayerNormLayer` | ✅ | ✅ **P** | ✅ GPT | ✅ |
| `GroupNormLayer` | ✅ | ✅ **P** | ✅ difusión y autoencoder | ✅ |
| `RMSNormLayer` | ✅ | ✅ **P** | ❌ **no está en el GPT** | — |
| `SwiGLU` | ✅ | ✅ **P** | ❌ **no está en el GPT** | — |
| `CrossAttention` | ✅ | ✅ **P** | ❌ **ningún modelo la usa aún** | — |
| `Upsample2D` y `Downsample2D` | ✅ | ✅ **P** | ✅ | ✅ |
| `ResidualBlock` | ✅ | ✅ | ⚠️ solo demo | — |
| `ResBlock2D` | ✅ | ✅ **P** | ✅ autoencoder | ✅ |
| `GraphConv` | ✅ | ✅ | ⚠️ solo demo | — |
| Activaciones (ReLU, GELU, SiLU, …) | ✅ | ✅ **P** | ✅ | ✅ |

**Las tres filas en rojo son las que más confusión causan.** `RMSNorm` y
`SwiGLU` son piezas del «transformer moderno» que se construyeron y verificaron,
pero el `GPTModel` sigue usando `LayerNorm` y un MLP con GELU. `CrossAttention`
se construyó para el puente texto→imagen, que todavía no existe.

---

## Modelos y generación

| Componente | Implementado | Verificado | Integrado | Entrenado |
| --- | :---: | :---: | :---: | :---: |
| `GPTModel` | ✅ | ✅ **P** | ✅ | ✅ `es_base` |
| `CRNNModel` (OCR) | ✅ | ✅ **P** | ✅ | ✅ |
| `DiffusionSchedule` | ✅ | ✅ **P** | ✅ | ✅ |
| Embedding sinusoidal del tiempo | ✅ | ✅ **P** | ✅ | ✅ |
| `ResBlockTiempo` | ✅ | ✅ **P** | ✅ | ✅ |
| `UNet2D` | ✅ | ✅ **P** | ✅ | ✅ `unet_mnist` |
| `DDPMSampler` | ✅ | ✅ **P** | ✅ | n/a |
| `DDIMSampler` | ✅ | ✅ **P** | ✅ | n/a |
| `Codificador` y `Decodificador` | ✅ | ✅ **P** | ✅ | ✅ `ae_c4` |
| `GaussianaDiagonal` (latente y KL) | ✅ | ✅ **P** | ✅ | ✅ |
| Difusión **latente** (LDM-2) | ❌ | — | — | — |

Los tres modelos entrenados, con sus datos y resultados, están en
[MODELOS.md](MODELOS.md).

---

## Imagen y datos

| Componente | Implementado | Verificado | Integrado |
| --- | :---: | :---: | :---: |
| Decodificador PNG (con `inflate` propio) | ✅ | ✅ **Pillow** | ✅ |
| Decodificador JPEG (línea base y progresivo) | ✅ | ✅ **Pillow** | ✅ |
| Decodificadores BMP y Netpbm | ✅ | ✅ **Pillow** | ✅ |
| Codificador PNG (gris) | ✅ | ✅ **Pillow** | ✅ muestreo |
| Enderezar, binarizar y cortar renglones | ✅ | ✅ | ✅ OCR |

---

## Lo que **no** está implementado, y por qué

| | Estado | Motivo |
| --- | --- | --- |
| **Difusión latente (LDM-2)** | Siguiente fase | Todo lo que necesita ya existe |
| **FID** | Pendiente, Fase 19 | Se hará con un clasificador propio; importar Inception rompería la premisa |
| **BPE** | Pendiente, Fase 08 | El tokenizador de caracteres funciona; el BPE acortaría las secuencias |
| **CTC** | No implementado | El corpus de OCR se genera con la alineación conocida |
| **GQA** | **Descartado con medición** | Ahorraría 0.79 MB y recortaría capacidad con solo 4 cabezas |
| **`Dropout`** | No implementado | Ningún modelo lo ha necesitado |
| **`dtype` distinto de `float`** | Pendiente, Fase 13 | Ningún caso de uso lo pide |
| **Pérdida perceptual (VGG)** | Decisión abierta | Necesitaría pesos preentrenados ajenos |
| **Condicionar por texto** | Horizonte | Necesita el puente `CrossAttention` + Transformer |
| **GPU, BF16, multi-máquina** | Decisión aplazada | Se abordará cuando un experimento interesante tarde días |

---

## Qué significa aquí «verificado»

Una **P** no es un adorno: significa que esa pieza se comparó contra PyTorch con
los mismos pesos y las mismas entradas, y que la diferencia quedó por debajo de
la tolerancia. Es lo que distingue «mi backward deriva mi forward» de «mi forward
es la arquitectura que dice ser».

Los casos de paridad actuales son cinco —`gpt`, `lstm`, `bilstm`, `crnn` y
`bloques`—, más la comparación de imagen contra Pillow. `bloques` es el que cubre
casi todo lo nuevo: RMSNorm, SiLU, GroupNorm, remuestreo, CrossAttention, SwiGLU,
RoPE, el calendario de difusión, el embedding del tiempo, los dos bloques
residuales, la `UNet2D`, DDPM y DDIM con y sin recorte, el codificador, el
decodificador y el latente gaussiano.
