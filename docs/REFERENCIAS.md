# Referencias bibliográficas

Cada pieza de NeuralSuite implementa un artículo concreto. Esta tabla dice cuál,
dónde vive el código y **en qué nos apartamos del original**, que suele ser lo
más útil de saber.

Las desviaciones no son descuidos: casi todas son decisiones tomadas por lo que
cabe en una CPU o por la premisa del proyecto de no depender de pesos ajenos.
Cuando una decisión salió de una medición, el detalle está en
[ROADMAP.md](ROADMAP.md).

---

## Fundamentos

| Componente | Código | Referencia |
| --- | --- | --- |
| Retropropagación | todo el framework | Rumelhart, Hinton y Williams (1986). *Learning representations by back-propagating errors*. Nature 323 |
| Adaline / regla delta | `demos/demo_adaline.cpp` | Widrow y Hoff (1960). *Adaptive switching circuits*. IRE WESCON |
| Inicialización Xavier | `Tensor::XavierInit`, usada por `Linear`, `Conv2D`, `LSTM` y la atención | Glorot y Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS |
| Comprobación por diferencias finitas | `tests/test_suite.cpp` | Práctica estándar; ver [VERIFICACION.md](VERIFICACION.md) |

## Optimización

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Adam | `optimizers.h` | Kingma y Ba (2014), arXiv:1412.6980 | — |
| AdamW (decaimiento desacoplado) | `optimizers.h` | Loshchilov y Hutter (2017), arXiv:1711.05101 | — |
| Tasa con coseno | `entrenamiento/checkpoint.h` | Loshchilov y Hutter (2016), *SGDR*, arXiv:1608.03983 | Solo el decaimiento coseno, sin los reinicios cálidos del artículo |
| Media móvil de pesos (EMA) | `optimizers.h` | Polyak y Juditsky (1992), SIAM J. Control Optim. 30(4); su uso en difusión, Ho et al. (2020) | Rampa de calentamiento `(1+n)/(10+n)`, como las implementaciones de referencia |

## Redes densas, convolucionales y recurrentes

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Convolución para imagen | `layers/conv2d.h` | LeCun et al. (1998). *Gradient-based learning applied to document recognition*. Proc. IEEE 86(11) | — |
| Convolución por `im2col` + GEMM | `src/layers/conv2d.cpp` | Chellapilla, Puri y Simard (2006). *High Performance Convolutional Neural Networks for Document Processing* | Se conserva la versión literal (`Conv2DReference`) para verificar la rápida |
| Bloques residuales | `layers/residual.h`, `layers/resblock2d.h` | He et al. (2015). *Deep Residual Learning*, arXiv:1512.03385 | — |
| LSTM | `layers/lstm.h` | Hochreiter y Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation 9(8) | — |
| LSTM bidireccional | `layers/lstm.h` (`BiLSTM`) | Graves y Schmidhuber (2005). *Framewise phoneme classification with bidirectional LSTM*. Neural Networks 18 | — |
| Convolución sobre grafos | `layers/graph_conv.h` | Kipf y Welling (2016). *Semi-Supervised Classification with GCN*, arXiv:1609.02907 | — |

## Transformer y modelo de lenguaje

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Atención multi-cabeza, codificación posicional sinusoidal | `layers/attention.h`, `gpt.h` | Vaswani et al. (2017). *Attention Is All You Need*, arXiv:1706.03762 | — |
| Decodificador solo (GPT) | `gpt.h` | Radford et al. (2018, 2019). *Improving Language Understanding by Generative Pre-Training*; *Language Models are Unsupervised Multitask Learners* | Tokenizador de caracteres, no BPE |
| Normalización por capa | `layers/layernorm.h` | Ba, Kiros y Hinton (2016), arXiv:1607.06450 | — |
| GELU | `activations.h` | Hendrycks y Gimpel (2016), arXiv:1606.08415 | — |
| Pesos compartidos entre embedding y salida | `src/gpt.cpp` | Press y Wolf (2016), arXiv:1608.05859 | — |
| RMSNorm | `layers/rmsnorm.h` | Zhang y Sennrich (2019), arXiv:1910.07467 | Verificada contra PyTorch, **aún no integrada en el GPT** |
| SwiGLU | `layers/swiglu.h` | Shazeer (2020). *GLU Variants Improve Transformer*, arXiv:2002.05202 | Igual: verificada, no integrada |
| RoPE | `layers/attention.h`, `tensor.h` | Su et al. (2021). *RoFormer*, arXiv:2104.09864 | Rota pares **adyacentes** (convención del artículo), no la mitad-y-mitad de LLaMA |
| KV-Cache | `layers/attention.h` | Técnica estándar de inferencia; descrita en Shazeer (2019), arXiv:1911.02150 | Ventana deslizante: con RoPE desaloja en vez de reconstruir |
| GQA | **no implementado** | Ainslie et al. (2023), arXiv:2305.13245 | Descartado con medición: ahorraría 0.79 MB y recortaría capacidad; ver Fase 15 |
| BPE | **pendiente** | Sennrich, Haddow y Birch (2015), arXiv:1508.07909 | Fase 08 |

## OCR

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| CRNN (convolución + recurrente para leer texto) | `models/ocr.h` | Shi, Bai y Yao (2015). *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*, arXiv:1507.05717 | **Sin CTC.** El corpus se genera con la alineación conocida, así que se entrena con entropía cruzada por columna; el colapso de repeticiones se hace al decodificar |
| CTC | **no implementado** | Graves et al. (2006). *Connectionist Temporal Classification*. ICML | Sería necesario con datos reales sin alineación |

## Generativos

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Autoencoder variacional: reparametrización y KL | `latent/gaussiana.h` | Kingma y Welling (2013). *Auto-Encoding Variational Bayes*, arXiv:1312.6114 | — |
| GAN | `demos/demo_gan.cpp` | Goodfellow et al. (2014), arXiv:1406.2661 | Demostración; todavía no es el discriminador del autoencoder |
| Difusión (DDPM) | `diffusion/schedule.h`, `diffusion/sampler.h` | Ho, Jain y Abbeel (2020). *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239 | Varianza posterior, no `beta` a secas |
| Muestreo acelerado (DDIM) | `diffusion/sampler.h` | Song, Meng y Ermon (2020). *Denoising Diffusion Implicit Models*, arXiv:2010.02502 | Al recortar `x₀` se **recalcula** `eps`, como en las implementaciones de referencia; sin eso, empeoraba al añadir pasos |
| U-Net | `diffusion/unet.h` | Ronneberger, Fischer y Brox (2015), arXiv:1505.04597 | **Sin capas de atención**: el DDPM original las pone a 16×16, aquí no hay ninguna |
| GroupNorm | `layers/groupnorm.h` | Wu y He (2018). *Group Normalization*, arXiv:1803.08494 | — |
| SiLU / Swish | `activations.h` | Elfwing, Uchibe y Doya (2017), arXiv:1702.03118; Ramachandran, Zoph y Le (2017), arXiv:1710.05941 | — |
| Ampliar por vecino más próximo y convolucionar | `layers/resample2d.h`, `latent/autoencoder.h` | Odena, Dumoulin y Olah (2016). *Deconvolution and Checkerboard Artifacts*. Distill | Se evita la convolución transpuesta por el patrón de tablero |
| Difusión latente | `latent/autoencoder.h` | Rombach et al. (2021). *High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv:2112.10752 | Pérdida **cuadrática**, sin la perceptual ni el discriminador por parches del artículo; se sella **media y escala** del latente, no solo la escala |

## Evaluación

| Componente | Estado | Referencia | Nota |
| --- | --- | --- | --- |
| FID | **pendiente**, Fase 19 | Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule*, arXiv:1706.08500 | Se calculará con un clasificador MNIST **entrenado aquí**, no con Inception: importar pesos ajenos rompería la premisa del proyecto |
| Perplejidad | `apps/eval_llm.cpp` | Medida estándar en modelado de lenguaje | Sobre un autor apartado del entrenamiento |
| PSNR | `apps/train_autoencoder.cpp` | Medida estándar en compresión de imagen | Siempre acompañada de un control, porque por sí sola engaña |

## Formatos y datos

| Componente | Código | Referencia |
| --- | --- | --- |
| PNG y su `inflate` | `image/png.h`, `image/inflate.h` | RFC 2083 (PNG), RFC 1950 (zlib), RFC 1951 (DEFLATE) |
| JPEG (línea base y progresivo) | `image/jpeg.h` | ITU-T T.81 / ISO-IEC 10918-1 |
| BMP y Netpbm | `image/bmp.h`, `image/netpbm.h` | Especificaciones de los formatos |
| MNIST | `data/mnist.h` | LeCun, Cortes y Burges. *The MNIST database of handwritten digits* |
| Corpus en español | `tools/corpus/` | Obras de dominio público de Project Gutenberg |

---

## Lo que el proyecto **no** toma de ningún artículo

Tres cosas que suelen darse por sentadas y aquí se decidieron a propósito:

- **No hay `Dropout`.** Ningún modelo del proyecto lo ha necesitado todavía.
- **No se importan pesos preentrenados de ningún sitio.** Ni Inception para FID ni
  VGG para la pérdida perceptual. Es la restricción que define el proyecto, y la
  razón de que algunas métricas se calculen de otra forma.
- **La verificación contra PyTorch** no viene de ningún artículo: es la práctica
  de comparar contra una implementación de referencia, descrita en
  [VERIFICACION.md](VERIFICACION.md).

## Para leer el fondo matemático

El repositorio incluye derivaciones propias, no resúmenes de los artículos:

- [DOCS_MATHEMATICS.md](../DOCS_MATHEMATICS.md): el Transformer paso a paso.
