# Referencias: de dónde sale cada cosa y dónde estudiarla

Cuatro cosas, en este orden:

1. **[Los papers fundamentales](#papers-fundamentales)**: los diecisiete
   artículos sobre los que se sostiene el proyecto, con qué introdujo cada uno,
   dónde vive en el código y en qué estado está.
2. **[Rutas de estudio](#rutas-de-estudio)**: cuatro itinerarios cerrados que
   combinan lectura, código de este repositorio y la prueba que lo verifica.
3. **[La biblioteca](#la-biblioteca)**: libros, cursos, explicaciones en línea e
   implementaciones de referencia, con qué cubre cada uno de lo que hay aquí.
4. **[La trazabilidad, pieza a pieza](#trazabilidad-pieza-a-pieza)**: qué
   artículo implementa cada componente y **en qué nos apartamos de él**. Es lo
   que consultar para auditar, no para aprender.

Casi todo lo enlazado es **gratuito y legal**: los autores publican sus libros en
abierto. Cuando algo solo existe en papel, se da el ISBN y se dice por qué vale
la pena.

---

---

# Papers fundamentales

Si la pregunta es «¿en qué se sostiene todo esto?», la respuesta son estos
diecisiete artículos. Cada fila dice qué introdujo, dónde vive en el código y en
qué estado está, con el mismo criterio que [ESTADO.md](ESTADO.md): **P** significa
paridad comprobada contra PyTorch.

| Año | Artículo | Qué introdujo | Dónde está aquí | Estado |
| --- | --- | --- | --- | --- |
| 1986 | Rumelhart, Hinton y Williams. *Learning representations by back-propagating errors* | La retropropagación | Todo el framework | Base de todo; cada capa se comprueba por diferencias finitas |
| 1997 | Hochreiter y Schmidhuber. *Long Short-Term Memory* | Memoria recurrente con puertas | `include/layers/lstm.h` | ✅ **P**, entrenada en el OCR |
| 1998 | LeCun et al. *Gradient-based learning applied to document recognition* | La convolución aplicada a visión | `include/layers/conv2d.h` | ✅ **P**, entrenada |
| 2013 | Kingma y Welling. *[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)* | Latentes probabilísticos y reparametrización | `include/latent/gaussiana.h` | ✅ **P**, entrenada en `ae_c4` |
| 2014 | Goodfellow et al. *[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)* | Generación adversarial | `demos/demo_gan.cpp` | ⚠️ solo demostración; no es aún el discriminador del autoencoder |
| 2014 | Kingma y Ba. *[Adam](https://arxiv.org/abs/1412.6980)* | Optimización adaptativa | `include/optimizers.h` | ✅ es el optimizador de los tres modelos |
| 2015 | He et al. *[Deep Residual Learning](https://arxiv.org/abs/1512.03385)* | Conexiones residuales | `include/layers/resblock2d.h` | ✅ **P**, entrenada en el autoencoder |
| 2015 | Ronneberger, Fischer y Brox. *[U-Net](https://arxiv.org/abs/1505.04597)* | El esqueleto de la difusión: bajar, subir y saltos | `include/diffusion/unet.h` | ✅ **P**, entrenada; omite la **self-attention a 16×16** que añadió la U-Net de Ho et al., no el artículo original |
| 2015 | Shi, Bai y Yao. *[CRNN](https://arxiv.org/abs/1507.05717)* | Leer texto de una imagen | `include/models/ocr.h` | ✅ **P**, entrenada; **sin CTC** |
| 2016 | Ba, Kiros y Hinton. *[Layer Normalization](https://arxiv.org/abs/1607.06450)* | La normalización del Transformer | `include/layers/layernorm.h` | ✅ **P**, integrada en el GPT |
| **2017** | **Vaswani et al. *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*** | **La atención multi-cabeza: el nacimiento de la arquitectura** | `include/layers/attention.h`, `include/gpt.h` | ✅ **P**, entrenada en `es_base` |
| 2018–19 | Radford et al. *Improving Language Understanding by Generative Pre-Training*; *Language Models are Unsupervised Multitask Learners* | El decodificador solo, autorregresivo | `include/gpt.h` | ✅ **P**, entrenado; tokenizador de caracteres, no BPE |
| 2018 | Wu y He. *[Group Normalization](https://arxiv.org/abs/1803.08494)* | Normalizar sin depender del lote | `include/layers/groupnorm.h` | ✅ **P**, entrenada |
| 2020 | Ho, Jain y Abbeel. *[DDPM](https://arxiv.org/abs/2006.11239)* | La difusión como modelo generativo | `include/diffusion/` | ✅ **P**, entrenada en `unet_mnist` |
| 2020 | Song, Meng y Ermon. *[DDIM](https://arxiv.org/abs/2010.02502)* | Muestreo acelerado y determinista | `include/diffusion/sampler.h` | ✅ **P**; 100 pasos en vez de 1000 |
| 2021 | Su et al. *[RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)* | La posición por rotación | `include/layers/attention.h`, `include/tensor.h` | ⚠️ **P** e integrado con `--rope`, pero **ningún modelo entrenado lo usa** |
| 2021 | Rombach et al. *[Latent Diffusion](https://arxiv.org/abs/2112.10752)* | Difundir en el latente, no en los píxeles | `include/latent/autoencoder.h` | ⏳ autoencoder hecho y medido (Fase 18); **generar sobre el latente es la Fase 19** |

Tres de esas filas no son verdes, y esa es justo la información que un índice de
referencias no suele dar: la GAN es una demostración aislada, RoPE está
implementado pero sin entrenar, y de la difusión latente existe la mitad.

# Rutas de estudio

Cuatro itinerarios cerrados. Cada paso dice **qué leer**, **qué mirar en el
repositorio** y **qué ejecutar** para comprobar que lo entendiste. Las pruebas se
lanzan con `./bin/test_suite`; cada una se anuncia con su número.

## Ruta 1 — Entender una red neuronal desde cero

Para quien empieza. Unas dos semanas a ritmo tranquilo.

1. Nielsen, capítulos 1 y 2, y el vídeo de [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).
2. Mira `include/tensor.h` y `include/layers/linear.h`: una capa es una matriz y
   un sesgo, nada más.
3. Lee `include/losses.h` y ejecuta `./bin/demo_mlp`.
4. Estudia cómo se comprueba un gradiente: busca `Test 4` en
   `tests/test_suite.cpp` —GELU contra diferencias finitas—. Ese patrón se
   repite en casi todas las 60 pruebas.
5. Cierra con [Ruder](https://www.ruder.io/optimizing-gradient-descent/) e
   `include/optimizers.h`.

## Ruta 2 — Del MLP al Transformer

Presupone la ruta 1.

1. [El Transformer ilustrado](https://jalammar.github.io/illustrated-transformer/).
2. [DOCS_MATHEMATICS.md](../DOCS_MATHEMATICS.md): la derivación propia, con las
   formas de cada matriz.
3. `include/layers/attention.h`, en este orden: `MultiHeadAttentionReference`
   —lenta y obvia— y después la rápida. Son el mismo cálculo.
4. `include/gpt.h`: bloque, residual, normalización y pesos compartidos.
5. [El Transformer anotado](https://nlp.seas.harvard.edu/annotated-transformer/)
   y [«Let's build GPT»](https://www.youtube.com/watch?v=kCc8FmEb1nY) para
   contrastar con otra implementación.
6. Entrena el tuyo con [GUIA_LLM.md](GUIA_LLM.md) y compáralo con lo que salió en
   [MODELOS.md](MODELOS.md).
7. Extra: RoPE, en Su et al. y en `RopeForward` / `RopeBackward`
   (`include/tensor.h`), y por qué aquí rota pares adyacentes. El gradiente es
   la misma rotación por el ángulo opuesto, que es una de las derivaciones más
   bonitas del repositorio.

## Ruta 3 — Generar imágenes

Presupone la ruta 1. Es la parte con más matemática del proyecto.

1. [Weng sobre difusión](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/),
   entera. Vuelve a ella cada vez que dudes.
2. `include/diffusion/schedule.h`: `x_t = √ᾱ·x₀ + √(1−ᾱ)·ε` es toda la parte
   directa, y está en diez líneas.
3. `include/diffusion/sampler.h`: DDPM y DDIM, con el detalle del recorte de
   `x₀`. Ese trozo existe por un defecto real; está contado en
   [VERIFICACION.md](VERIFICACION.md).
4. Ejecuta `Test 54`: los muestreadores contra un oráculo analítico.
5. [GUIA_DIFUSION.md](GUIA_DIFUSION.md), y genera imágenes con un modelo ya
   entrenado antes de entrenar ninguno.
6. Para el latente: [Weng sobre VAE](https://lilianweng.github.io/posts/2018-08-12-vae/),
   `include/latent/gaussiana.h` y [GUIA_AUTOENCODER.md](GUIA_AUTOENCODER.md).
7. Cierra con [Rombach et al.](https://arxiv.org/abs/2112.10752) y el
   [código de CompVis](https://github.com/CompVis/latent-diffusion), que es
   adonde apunta la Fase 19.

## Ruta 4 — Construir un framework, no solo usarlo

La que no cubre ningún curso.

1. [DOCS_PROGRAMMING_CPP.md](../DOCS_PROGRAMMING_CPP.md): los ocho patrones del
   código, cada uno con el defecto real que lo motivó.
2. [ARQUITECTURA.md](ARQUITECTURA.md), con los diagramas.
3. `include/parallel.h` y el capítulo de hilos de Williams: por qué el reparto es
   dinámico y por qué **ningún hilo reduce sobre otro**.
4. [VERIFICACION.md](VERIFICACION.md), completo. Es el documento más
   característico del proyecto: diferencias finitas, paridad, mutación y
   controles, con lo que encontró cada capa.
5. `tools/parity/`: exportar pesos, ejecutar las dos versiones, comparar.
   Reprodúcelo contra [LLMRasec](https://github.com/iztaneo/LLMRasec).
6. `include/serialization.h` y `include/entrenamiento/checkpoint.h`: qué hace
   falta guardar para que reanudar sea **idéntico bit a bit**. La respuesta
   incluye cosas que no son obvias, como el contador de pasos de la EMA.
7. Compara con [llm.c](https://github.com/karpathy/llm.c), que resuelve el mismo
   problema en C.

---

# La biblioteca

## Libros

### Para el fondo general

| Libro | Acceso | Qué cubre de lo que hay aquí |
| --- | --- | --- |
| Goodfellow, Bengio y Courville. **Deep Learning** (MIT Press, 2016) | [libre](https://www.deeplearningbook.org/) | La referencia canónica. Cap. 6 retropropagación, 8 optimización, 9 convolución, 10 recurrentes, 20 modelos generativos. Denso y sin código |
| Prince. **Understanding Deep Learning** (MIT Press, 2023) | [PDF libre](https://udlbook.github.io/udlbook/) | El sustituto moderno del anterior: incluye Transformers y difusión, con figuras excelentes. Si solo vas a leer un libro, este |
| Zhang, Lipton, Li y Smola. **Dive into Deep Learning** | [libre, con código](https://www.d2l.ai/) | Cada concepto con su implementación ejecutable. El más cercano al espíritu de NeuralSuite: [el capítulo de LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html) y [el de atención](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) explican exactamente lo que está en `include/layers/` |
| Bishop y Bishop. **Deep Learning: Foundations and Concepts** (2023) | [web del libro](https://www.bishopbook.com/) | Tiene un capítulo de difusión que deriva el DDPM con cuidado, cosa rara en un libro de texto |
| Fleuret. **The Little Book of Deep Learning** | [PDF libre](https://fleuret.org/francois/lbdl.html) | 160 páginas de móvil. Para repasar, no para aprender de cero |
| Nielsen. **Neural Networks and Deep Learning** | [libre](http://neuralnetworksanddeeplearning.com/) | Construye la retropropagación desde cero, a mano, con intuición. El mejor primer contacto si el cálculo matricial todavía incomoda |

### Para la matemática y la probabilidad

| Libro | Acceso | Para qué |
| --- | --- | --- |
| Deisenroth, Faisal y Ong. **Mathematics for Machine Learning** | [libre](https://mml-book.github.io/) | Álgebra lineal, cálculo vectorial y probabilidad, solo lo que hace falta. Empieza por aquí si las derivadas matriciales del `Backward` no se leen solas |
| Murphy. **Probabilistic Machine Learning: An Introduction** (2022) | [libre](https://probml.github.io/pml-book/book1.html) | El marco probabilista: máxima verosimilitud, que es de donde sale la entropía cruzada de `losses.h` |
| Murphy. **Probabilistic ML: Advanced Topics** (2023) | [libre](https://probml.github.io/pml-book/book2.html) | Los capítulos de VAE y de modelos de difusión, con la derivación del ELBO completa |
| MacKay. **Information Theory, Inference, and Learning Algorithms** | [libre](https://www.inference.org.uk/mackay/itila/) | Entropía, divergencia KL y **perplejidad**. Lo que hace falta para entender de verdad el número que imprime `eval_llm` |
| Bishop. **Pattern Recognition and Machine Learning** (2006) | [PDF libre](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) | Clásico. El capítulo 5 sigue siendo una de las mejores exposiciones de la retropropagación |
| Hardt y Recht. **Patterns, Predictions, and Actions** | [libre](https://mlstory.org/) | Por qué generalizar es posible. Contexto para lo que se mide en [MODELOS.md](MODELOS.md) |

### Para lenguaje

| Libro | Acceso | Para qué |
| --- | --- | --- |
| Jurafsky y Martin. **Speech and Language Processing**, 3.ª ed. | [borrador libre](https://web.stanford.edu/~jurafsky/slp3/) | Modelado de lenguaje, perplejidad, tokenización y BPE, y un capítulo de Transformers. La referencia del área |

### Para el C++

| Libro | Acceso | Para qué |
| --- | --- | --- |
| Meyers. **Effective Modern C++** (O'Reilly, 2014) | ISBN 978-1491903995 | Los 42 consejos sobre C++11/14 que explican por qué `Tensor` usa `shared_ptr` y movimiento como lo usa |
| Williams. **C++ Concurrency in Action**, 2.ª ed. (Manning, 2019) | ISBN 978-1617294693 | Hilos, `std::future` y modelos de memoria: el fondo de `include/parallel.h` |
| Stroustrup. **The C++ Programming Language**, 4.ª ed. | ISBN 978-0321563842 | La referencia del lenguaje, del autor |
| **C++ Core Guidelines** | [libre](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) | Las reglas que este código intenta seguir; ver [DOCS_PROGRAMMING_CPP.md](../DOCS_PROGRAMMING_CPP.md) |
| **cppreference** | [libre](https://en.cppreference.com/w/) | La referencia de la biblioteca estándar que se consulta a diario |

---

## Cursos

| Curso | Acceso | Qué aporta |
| --- | --- | --- |
| Karpathy. **Neural Networks: Zero to Hero** | [libre, en vídeo](https://karpathy.ai/zero-to-hero.html) | **El más cercano a este proyecto.** Construye un motor de autograd y luego un GPT, línea a línea, sin saltarse nada. Los dos imprescindibles: [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) y [«Let's build GPT»](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| Stanford **CS231n**, visión por computador | [apuntes libres](https://cs231n.github.io/) · [web](https://cs231n.stanford.edu/) | Los apuntes de [convolución](https://cs231n.github.io/convolutional-networks/) explican `im2col`, que es justo lo que hace `src/layers/conv2d.cpp`; los de [retropropagación](https://cs231n.github.io/optimization-2/) son el mejor texto corto sobre el tema |
| Stanford **CS224n**, lenguaje natural | [web](https://web.stanford.edu/class/cs224n/) | Del modelo de lenguaje a los Transformers. Diapositivas y vídeos abiertos |
| **fast.ai**, Practical Deep Learning | [libre](https://www.fast.ai/) | Enfoque de arriba abajo. Su curso de difusión desde cero es un buen complemento |
| **DeepLearning.AI** | [catálogo](https://www.deeplearning.ai/courses/) | Cursos cortos y estructurados; la especialización clásica de Ng sigue siendo un buen primer paso |

---

## Explicaciones en línea que valen más que el artículo

Un artículo se escribe para revisores, no para quien aprende. Estos textos
explican lo mismo mejor, y en varios casos son lo que hay que leer **antes** del
artículo.

| Tema | Enlace | Por qué |
| --- | --- | --- |
| Retropropagación a mano | [CS231n, backprop](https://cs231n.github.io/optimization-2/) | El grafo de cómputo y la regla de la cadena, con ejemplos numéricos |
| Optimizadores | [Ruder, *An overview of gradient descent optimization algorithms*](https://www.ruder.io/optimizing-gradient-descent/) | SGD, momento, RMSProp y Adam en una sola página coherente |
| LSTM | [Olah, *Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Los diagramas de puertas que todo el mundo cita. Léelo con `include/layers/lstm.h` al lado |
| Recurrentes a nivel de carácter | [Karpathy, *The Unreasonable Effectiveness of RNNs*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Explica exactamente lo que hace `es_base`: aprender el idioma carácter a carácter |
| Transformer | [Alammar, *The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) | La explicación visual estándar |
| Transformer con código | [*The Annotated Transformer* (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) | El artículo entero anotado con PyTorch ejecutable, párrafo a párrafo |
| Atención, paso a paso | [Raschka, *Self-attention from scratch*](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) | Las matrices Q, K y V con números concretos |
| Qué hace un Transformer por dentro | [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) | Para después: descompone la atención en circuitos interpretables |
| VAE | [Weng, *From Autoencoder to Beta-VAE*](https://lilianweng.github.io/posts/2018-08-12-vae/) | Reparametrización y ELBO, que es lo que implementa `latent/gaussiana.h` |
| Difusión | [Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | **La mejor referencia única de difusión.** Todas las fórmulas del muestreador salen de aquí |
| Difusión, con código | [*The Annotated Diffusion Model* (Hugging Face)](https://huggingface.co/blog/annotated-diffusion) | El DDPM implementado y comentado |
| Difusión, la derivación completa | [Luo, *Understanding Diffusion Models: A Unified Perspective*](https://arxiv.org/abs/2208.11970) | 40 páginas que derivan todo desde el ELBO sin saltos |
| Difusión como puntuación | [Song, *Generative Modeling by Estimating Gradients*](https://yang-song.net/blog/2021/score/) | La otra forma de ver lo mismo; aclara por qué predecir el ruido funciona |
| Guía y condicionamiento | [Dieleman, *Guidance: a cheat code for diffusion models*](https://sander.ai/2022/05/26/guidance.html) | Lo que haría falta para pedir «un tres»: hoy el modelo no está condicionado |
| Difusión latente | [Alammar, *The Illustrated Stable Diffusion*](https://jalammar.github.io/illustrated-stable-diffusion/) | El sistema completo al que apunta la Fase 19 |
| Artefactos de tablero | [Odena et al., Distill](https://distill.pub/2016/deconv-checkerboard/) | Interactivo. Es la razón de que aquí se amplíe y luego se convolucione, en vez de usar convolución transpuesta |
| Aritmética de convolución | [Dumoulin y Visin](https://arxiv.org/abs/1603.07285) | Relleno, paso y formas de salida, con animaciones |
| CTC | [Hannun, *Sequence Modeling with CTC*](https://distill.pub/2017/ctc/) | Lo que **no** está implementado en el OCR, y qué haría falta |
| Atención, panorama | [Weng, *Attention? Attention!*](https://lilianweng.github.io/posts/2018-06-24-attention/) | Historia y variantes, útil para situar RoPE y GQA |

---

## Implementaciones de referencia

Leer código ajeno al lado del propio es la forma más rápida de detectar una
diferencia de arquitectura. Estas son las que se consultaron al construir y
verificar NeuralSuite.

| Implementación | Enlace | Para qué sirve aquí |
| --- | --- | --- |
| **LLMRasec** | [repositorio](https://github.com/iztaneo/LLMRasec) | El oráculo de este proyecto: los mismos modelos en PyTorch. Es contra lo que se compara en [VERIFICACION.md](VERIFICACION.md) |
| **micrograd** | [repositorio](https://github.com/karpathy/micrograd) | 150 líneas de autograd. El mejor punto de partida para entender `include/autograd.h` |
| **minGPT** | [repositorio](https://github.com/karpathy/minGPT) | GPT legible. Comparar con `include/gpt.h` aclara qué es esencial y qué es ingeniería |
| **nanoGPT** | [repositorio](https://github.com/karpathy/nanoGPT) | La versión entrenable del anterior, con el bucle de entrenamiento y el recorte de gradiente |
| **llm.c** | [repositorio](https://github.com/karpathy/llm.c) | GPT-2 en C puro. El pariente más próximo a lo que hace NeuralSuite, y una buena vara para medir rendimiento |
| **PyTorch** | [código](https://github.com/pytorch/pytorch) · [artículo](https://arxiv.org/abs/1912.01703) | Cuando la paridad falla, la duda se resuelve en la documentación de la capa —por ejemplo [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html), cuyo orden de puertas hay que respetar— o en el fuente |
| **DDPM, el original** | [repositorio](https://github.com/hojonathanho/diffusion) | El código del artículo de Ho et al., en TensorFlow |
| **denoising-diffusion-pytorch** | [repositorio](https://github.com/lucidrains/denoising-diffusion-pytorch) | La reimplementación que casi todos usan. De aquí sale la convención de recortar `x₀` **y recalcular `eps`**, que costó un defecto real |
| **guided-diffusion** (OpenAI) | [repositorio](https://github.com/openai/guided-diffusion) | La U-Net con atención y las mejoras de muestreo |
| **latent-diffusion** (CompVis) | [repositorio](https://github.com/CompVis/latent-diffusion) | El sistema del artículo de Rombach: autoencoder, escala del latente y difusión encima |
| **labml.ai annotated implementations** | [web](https://nn.labml.ai/) | Decenas de artículos implementados y anotados línea a línea |

---

# Trazabilidad, pieza a pieza

Las desviaciones no son descuidos: casi todas son decisiones tomadas por lo que
cabe en una CPU o por la premisa del proyecto de no depender de pesos ajenos.
Cuando una decisión salió de una medición, el detalle está en
[el diario de fases](history/DIARIO_FASES.md).

## Fundamentos

| Componente | Código | Referencia |
| --- | --- | --- |
| Retropropagación | todo el framework | Rumelhart, Hinton y Williams (1986). *Learning representations by back-propagating errors*. Nature 323 |
| Panorama del área, por sus autores | — | LeCun, Bengio y Hinton (2015). *[Deep Learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)*. Nature 521 |
| Adaline / regla delta | `demos/demo_adaline.cpp` | Widrow y Hoff (1960). *Adaptive switching circuits*. IRE WESCON |
| Inicialización Xavier | `Tensor::XavierInit`, usada por `Linear`, `Conv2D`, `LSTM` y la atención | Glorot y Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS |
| Comprobación por diferencias finitas | `tests/test_suite.cpp` | Práctica estándar; ver [VERIFICACION.md](VERIFICACION.md) |

**Para estudiarlo:** Nielsen, capítulos 1 y 2, y luego
[CS231n backprop](https://cs231n.github.io/optimization-2/). El vídeo de
[micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) cierra el tema.

## Optimización

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Adam | `include/optimizers.h` | Kingma y Ba (2014), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) | — |
| AdamW (decaimiento desacoplado) | `include/optimizers.h` | Loshchilov y Hutter (2017), [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | — |
| Tasa con coseno | `include/entrenamiento/checkpoint.h` | Loshchilov y Hutter (2016), *SGDR*, [arXiv:1608.03983](https://arxiv.org/abs/1608.03983) | Solo el decaimiento coseno, sin los reinicios cálidos del artículo |
| Media móvil de pesos (EMA) | `include/optimizers.h` | Polyak y Juditsky (1992), SIAM J. Control Optim. 30(4); su uso en difusión, Ho et al. (2020) | Rampa de calentamiento `(1+n)/(10+n)`, como las implementaciones de referencia |

**Para estudiarlo:** [Ruder](https://www.ruder.io/optimizing-gradient-descent/)
primero; *Deep Learning*, capítulo 8, para el fondo.

## Redes densas, convolucionales y recurrentes

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Convolución para imagen | `include/layers/conv2d.h` | LeCun et al. (1998). *Gradient-based learning applied to document recognition*. Proc. IEEE 86(11) | — |
| Convolución por `im2col` + GEMM | `src/layers/conv2d.cpp` | Chellapilla, Puri y Simard (2006). *High Performance Convolutional Neural Networks for Document Processing* | Se conserva la versión literal (`Conv2DReference`) para verificar la rápida |
| Bloques residuales | `include/layers/residual.h`, `include/layers/resblock2d.h` | He et al. (2015). *Deep Residual Learning*, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | — |
| LSTM | `include/layers/lstm.h` | Hochreiter y Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation 9(8) | — |
| LSTM bidireccional | `include/layers/lstm.h` (`BiLSTM`) | Graves y Schmidhuber (2005). *Framewise phoneme classification with bidirectional LSTM*. Neural Networks 18 | — |
| Convolución sobre grafos | `include/layers/graph_conv.h` | Kipf y Welling (2016). *Semi-Supervised Classification with GCN*, [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) | — |

**Para estudiarlo:** [CS231n](https://cs231n.github.io/convolutional-networks/)
para convolución e `im2col`; [Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
y [D2L](https://d2l.ai/chapter_recurrent-modern/lstm.html) para las LSTM.

## Transformer y modelo de lenguaje

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Atención multi-cabeza, codificación posicional sinusoidal | `include/layers/attention.h`, `include/gpt.h` | Vaswani et al. (2017). *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | — |
| Decodificador solo (GPT) | `include/gpt.h` | Radford et al. (2018, 2019). *Improving Language Understanding by Generative Pre-Training*; *Language Models are Unsupervised Multitask Learners* | Tokenizador de caracteres, no BPE |
| Normalización por capa | `include/layers/layernorm.h` | Ba, Kiros y Hinton (2016), [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) | — |
| GELU | `include/activations.h` | Hendrycks y Gimpel (2016), [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | — |
| Pesos compartidos entre embedding y salida | `src/gpt.cpp` | Press y Wolf (2016), [arXiv:1608.05859](https://arxiv.org/abs/1608.05859) | — |
| RMSNorm | `include/layers/rmsnorm.h` | Zhang y Sennrich (2019), [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | Verificada contra PyTorch, **aún no integrada en el GPT**; ver [ESTADO.md](ESTADO.md) |
| SwiGLU | `include/layers/swiglu.h` | Shazeer (2020). *GLU Variants Improve Transformer*, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | Igual: verificada, no integrada |
| RoPE | `include/layers/attention.h`, `include/tensor.h` | Su et al. (2021). *RoFormer*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | Rota pares **adyacentes** (convención del artículo), no la mitad-y-mitad de LLaMA |
| KV-Cache | `include/layers/attention.h` | Técnica estándar de inferencia; descrita en Shazeer (2019), [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | Ventana deslizante: con RoPE desaloja en vez de reconstruir |
| GQA | **no implementado** | Ainslie et al. (2023), [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | Descartado con medición: ahorraría 0.79 MB y recortaría capacidad con solo 4 cabezas |
| BPE | **pendiente** | Sennrich, Haddow y Birch (2015), [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) | Fase 08 |

**Para estudiarlo:** [el Transformer ilustrado](https://jalammar.github.io/illustrated-transformer/),
luego [el anotado](https://nlp.seas.harvard.edu/annotated-transformer/) con el
código delante, y [«Let's build GPT»](https://www.youtube.com/watch?v=kCc8FmEb1nY)
para verlo construir entero. La derivación propia está en
[DOCS_MATHEMATICS.md](../DOCS_MATHEMATICS.md).

## OCR

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| CRNN (convolución + recurrente para leer texto) | `include/models/ocr.h` | Shi, Bai y Yao (2015), [arXiv:1507.05717](https://arxiv.org/abs/1507.05717) | **Sin CTC.** El corpus se genera con la alineación conocida, así que se entrena con entropía cruzada por columna; el colapso de repeticiones se hace al decodificar |
| CTC | **no implementado** | Graves et al. (2006). *Connectionist Temporal Classification*. ICML; la [tesis de Graves](https://www.cs.toronto.edu/~graves/preprint.pdf) lo desarrolla | Sería necesario con datos reales sin alineación |

**Para estudiarlo:** [el artículo de Distill sobre CTC](https://distill.pub/2017/ctc/)
explica en media hora lo que aquí se evitó y por qué.

## Generativos

| Componente | Código | Referencia | Desviación |
| --- | --- | --- | --- |
| Autoencoder variacional: reparametrización y KL | `include/latent/gaussiana.h` | Kingma y Welling (2013). *Auto-Encoding Variational Bayes*, [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | — |
| GAN | `demos/demo_gan.cpp` | Goodfellow et al. (2014), [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) | Demostración; todavía no es el discriminador del autoencoder |
| Difusión (DDPM) | `include/diffusion/schedule.h`, `include/diffusion/sampler.h` | Ho, Jain y Abbeel (2020), [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | Varianza posterior, no `beta` a secas |
| Muestreo acelerado (DDIM) | `include/diffusion/sampler.h` | Song, Meng y Ermon (2020), [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) | Al recortar `x₀` se **recalcula** `eps`, como en las implementaciones de referencia; sin eso, empeoraba al añadir pasos |
| U-Net | `include/diffusion/unet.h` | Ronneberger, Fischer y Brox (2015), [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | El artículo de 2015 **no lleva atención**: es un camino contractivo, uno expansivo y los saltos. La que sí la lleva —self-attention a 16×16— es la U-Net que Ho et al. usan en el DDPM, y esa capa aquí no está |
| GroupNorm | `include/layers/groupnorm.h` | Wu y He (2018), [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) | — |
| SiLU / Swish | `include/activations.h` | Elfwing, Uchibe y Doya (2017), [arXiv:1702.03118](https://arxiv.org/abs/1702.03118); Ramachandran, Zoph y Le (2017), [arXiv:1710.05941](https://arxiv.org/abs/1710.05941) | — |
| Ampliar por vecino más próximo y convolucionar | `include/layers/resample2d.h`, `include/latent/autoencoder.h` | Odena, Dumoulin y Olah (2016). *[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)*. Distill | Se evita la convolución transpuesta por el patrón de tablero |
| Difusión latente | `include/latent/autoencoder.h` | Rombach et al. (2021), [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | Pérdida **cuadrática**, sin la perceptual ni el discriminador por parches del artículo; se sella **media y escala** del latente, no solo la escala |

**Para estudiarlo:** el orden que funciona es
[VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) →
[difusión](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) →
[el DDPM anotado](https://huggingface.co/blog/annotated-diffusion) → el artículo
original. Si quieres la derivación sin saltos,
[Luo (2022)](https://arxiv.org/abs/2208.11970).

## Evaluación

| Componente | Estado | Referencia | Nota |
| --- | --- | --- | --- |
| FID | **pendiente**, Fase 19 | Heusel et al. (2017), [arXiv:1706.08500](https://arxiv.org/abs/1706.08500) | Se calculará con un clasificador MNIST **entrenado aquí**, no con Inception: importar pesos ajenos rompería la premisa del proyecto |
| Perplejidad | `apps/eval_llm.cpp` | Medida estándar; su fundamento, en MacKay y en Jurafsky y Martin | Sobre un autor apartado del entrenamiento |
| PSNR | `apps/train_autoencoder.cpp` | Medida estándar en compresión de imagen | Siempre acompañada de un control, porque por sí sola engaña |

## Formatos y datos

| Componente | Código | Referencia |
| --- | --- | --- |
| PNG y su `inflate` | `include/image/png.h`, `include/image/inflate.h` | [RFC 2083](https://www.rfc-editor.org/rfc/rfc2083) (PNG), [RFC 1950](https://www.rfc-editor.org/rfc/rfc1950) (zlib), [RFC 1951](https://www.rfc-editor.org/rfc/rfc1951) (DEFLATE); la norma vigente es [W3C PNG 3.ª ed.](https://www.w3.org/TR/png/) |
| JPEG (línea base y progresivo) | `include/image/jpeg.h` | [ITU-T T.81](https://www.itu.int/rec/T-REC-T.81) / ISO-IEC 10918-1 |
| BMP y Netpbm | `include/image/bmp.h`, `include/image/netpbm.h` | Especificaciones de los formatos |
| MNIST | `include/data/mnist.h` | [LeCun, Cortes y Burges](http://yann.lecun.com/exdb/mnist/) |
| Corpus en español | `tools/corpus/` | Obras de dominio público de [Project Gutenberg](https://www.gutenberg.org/) |

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
