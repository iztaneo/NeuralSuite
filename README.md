# 🧠 NeuralSuite: C++17 Deep Learning Framework by KhepriNova

*NeuralSuite is an independent C++17 neural-network framework for transparent training, inference and experimentation, with no external ML or linear-algebra runtime dependencies.*

[![CI](https://github.com/iztaneo/NeuralSuite/actions/workflows/ci.yml/badge.svg)](https://github.com/iztaneo/NeuralSuite/actions/workflows/ci.yml)

**NeuralSuite** es un framework de aprendizaje profundo escrito en **C++17**, sin
PyTorch, TensorFlow, BLAS, Eigen ni ninguna otra biblioteca externa de IA o de
álgebra lineal: solo la biblioteca estándar. Entrena, evalúa e infiere modelos
reales, y cada pieza se verifica contra PyTorch.

Es **multiplataforma (Linux, macOS y Windows)** y paraleliza en CPU con
`std::thread`, sin OpenMP.

---

## Qué hace hoy

| Área | Qué se ha demostrado |
| --- | --- |
| **Lenguaje** | Un GPT entrenado sobre un corpus en español genera texto reconocible, con perplejidad **6.23** sobre un autor que nunca vio. Admite RoPE y genera con KV-Cache de ventana deslizante. |
| **Generación de imágenes** | Un modelo de difusión con U-Net, entrenado 80 000 iteraciones sobre MNIST, **genera dígitos nuevos** que no son copias del conjunto, con DDPM y DDIM. |
| **Compresión** | El autoencoder de la difusión latente reconstruye dígitos nunca vistos a **33.2 dB**, 13.8 dB por encima de un compresor simple del mismo tamaño; difundir sobre su latente cuesta **6.3× menos** que sobre los píxeles. |
| **Visión** | OCR de texto impreso con su propio decodificador de imagen: **3.4 %** de error de carácter en una página de libro (Tesseract: 0.1 %). No lee manuscrito, no distingue texto de dibujo y no maneja varias columnas; ver [tools/ocr/README.md](tools/ocr/README.md). |
| **Reproducibilidad** | Reanudar un entrenamiento desde un checkpoint da **pesos idénticos bit a bit** a no haberlo parado. |

**Estado: experimental, versión 0.x.** Es un framework serio para estudiar y
reproducir ideas de papers con verificación completa, pero **no es software de
producción**: no hay API estable, ni aceleración por GPU, ni evidencia más allá de
modelos pequeños y MNIST.

---

## Documentación

El índice completo está en **[docs/](docs/README.md)**. Los principales:

**Para probarlo en diez minutos**

- **[Guía rápida](docs/QUICKSTART.md)**: compilar, ejecutar las pruebas y generar tu primera imagen.

**Para entender el proyecto**

- **[Mapa del código](docs/MAPA_DEL_CODIGO.md)**: qué hay en cada carpeta y en qué orden leerlo.
- **[Arquitectura](docs/ARQUITECTURA.md)**: cómo encaja todo, con diagramas.
- **[Glosario](docs/GLOSARIO.md)**: el vocabulario del código y de los logs, de `shape` a `ᾱ`, explicado una vez.

**Para usarlo**

- **[Modelo de lenguaje](docs/GUIA_LLM.md)**: preparar el corpus, entrenar, evaluar y generar texto.
- **[Difusión](docs/GUIA_DIFUSION.md)**: entrenar la U-Net sobre MNIST, reanudar checkpoints y generar imágenes.
- **[Autoencoder (difusión latente)](docs/GUIA_AUTOENCODER.md)**: comprimir a un latente 8×8 y por qué difundir ahí cuesta 6 veces menos.
- **[Los modelos entrenados](docs/MODELOS.md)**: datos, comandos, resultados medidos y límites de cada uno.

**Para estudiar**

- **[Teoría e implementación](docs/TEORIA_E_IMPLEMENTACION.md)**: cada concepto con su matemática, su sitio en el código, cómo se verifica y de dónde sale.
- **[Referencias](docs/REFERENCIAS.md)**: los papers fundamentales con su sitio en el código, cuatro rutas de estudio, libros y cursos, y la trazabilidad pieza a pieza.
- **[Cómo se verifica](docs/VERIFICACION.md)**: paridad, diferencias finitas, mutaciones y controles, con los fallos reales que encontró cada capa.

**Para saber qué hay hecho y qué falta**

- **[Estado por componente](docs/ESTADO.md)**: qué existe, qué está verificado, qué está integrado en un modelo y qué ha entrenado de verdad.
- **[Hoja de ruta](docs/ROADMAP.md)**: lo que falta y en qué orden. El historial fase a fase está en [el diario](docs/history/DIARIO_FASES.md).

---

## Compilación

### Linux y macOS (Makefile)

```bash
make
```

Los ejecutables quedan en `bin/`. Para compilar y lanzar las pruebas:

```bash
make test
```

### Windows, Linux y macOS (CMake)

```bash
cmake -B build
cmake --build build
```

Con CMake los ejecutables quedan en `build/`. El build por defecto es `Release` y
portable (sin `-march=native`). Para desarrollo conviene el build con
comprobaciones, que valida los índices de `Tensor::operator[]`:

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
```

| Opción | Por defecto | Efecto |
| --- | --- | --- |
| `NEURALSUITE_NATIVE_ARCH` | `OFF` | `-march=native`: más rápido en esta CPU, no portable |
| `NEURALSUITE_FAST_MATH` | `OFF` | `-ffast-math`: relaja IEEE-754 y estorba al verificar |

---

## Ejecutables

**Aplicaciones** (`apps/`)

| Programa | Qué hace |
| --- | --- |
| `train_llm` | Entrena el GPT sobre un archivo de texto |
| `eval_llm` | Pérdida y perplejidad sobre train, val y test |
| `generate_llm` | Genera texto a partir de un prompt |
| `train_diffusion` | Entrena la U-Net de difusión sobre MNIST, con checkpoints sellados y reanudación |
| `sample_diffusion` | Genera imágenes con DDPM o DDIM y las guarda en PNG |
| `train_autoencoder` | Entrena el autoencoder de la difusión latente |
| `train_ocr` | Entrena el modelo de OCR |
| `ocr_cli` | Transcribe una imagen (PNG, JPEG, BMP o Netpbm) |

Todos aceptan `--help` o documentan sus opciones en la guía correspondiente.

**Pruebas**

- `./bin/test_suite`: 60 pruebas. Gradientes contra diferencias finitas,
  casos con respuesta exacta, formatos de archivo, infraestructura de checkpoint
  y validación de entradas. Termina con código distinto de cero si algo falla.

**Demostraciones** (`demos/`), una por técnica: `demo_mlp`, `demo_adaline`,
`demo_cnn`, `demo_lstm`, `demo_sequential`, `demo_resnet`, `demo_gnn`,
`demo_autoencoder`, `demo_gan`, `demo_diffusion`, `demo_autograd` y `demo_ocr`.

---

## Qué contiene

**Núcleo:** `Tensor` con vistas sin copia, `Parameter` y `Module`, un motor de
diferenciación automática (`autograd.h`), optimizadores `AdamW` y `SGD`, media
exponencial de pesos (`EMA`), pérdidas, serialización NSF y paralelismo con
resultado idéntico bit a bit para cualquier número de hilos.

**Capas:** `Linear`, `Conv2D`, `MaxPool2D`, `LSTM` y `BiLSTM`, `Embedding`,
`MultiHeadAttention` (con RoPE y KV-Cache), `CrossAttention`, `LayerNormLayer`,
`RMSNorm`, `GroupNorm`, `SwiGLU`, `Upsample2D`, `Downsample2D`, `ResidualBlock`,
`ResBlock2D` y `GraphConv`. `RMSNorm` y `SwiGLU` están verificadas contra PyTorch
pero todavía no forman parte del `GPTModel`.

**Modelos:** `GPTModel`, `CRNNModel` (OCR), `UNet2D` condicionada por el paso de
difusión, y el `Codificador`, `Decodificador` y latente gaussiano del autoencoder.

**Difusión:** calendario de ruido, embedding sinusoidal del tiempo, muestreadores
`DDPMSampler` y `DDIMSampler`.

**Datos:** lector de MNIST, `DataLoader` con particiones que comparten memoria,
tokenizadores de caracteres y de bytes, y decodificadores de PNG, JPEG, BMP y
Netpbm sin bibliotecas externas.

**Entrenamiento:** checkpoints de varios archivos escritos de forma transaccional,
sellados con toda la configuración del run, reanudación exacta y tasa con
calentamiento y coseno (`entrenamiento/checkpoint.h`).

---

## Verificación

Cada push a `main` ejecuta en [integración continua](.github/workflows/ci.yml):
compilación y pruebas en Linux (GCC y Clang), macOS y Windows, en `Debug` y
`Release`; la suite bajo AddressSanitizer y UndefinedBehaviorSanitizer; las demos;
un entrenamiento de extremo a extremo; y la **comparación contra PyTorch**.

Esa comparación usa la implementación de referencia
[LLMRasec](https://github.com/iztaneo/LLMRasec), con los mismos pesos y las mismas
entradas, y cubre el GPT, las LSTM, el OCR y todos los bloques de la difusión y el
autoencoder. Detecta lo que las diferencias finitas no pueden: que una
arquitectura sea coherente pero no la que se pretendía. Todo el detalle está en
[docs/VERIFICACION.md](docs/VERIFICACION.md).

---

## Estructura del repositorio

```
include/          Cabeceras: la interfaz y el porqué del diseño
  layers/           Capas
  models/           CRNN (OCR); el GPT está en `gpt.h`
  diffusion/        Calendario, U-Net y muestreadores
  latent/           Autoencoder de la difusión latente
  entrenamiento/    Checkpoints y reanudación
  data/             MNIST y DataLoader
  image/            Decodificadores y preproceso de imagen
src/              Implementaciones, con las mismas subcarpetas
apps/             Las ocho aplicaciones
demos/            Doce demostraciones, una por técnica
tests/            La suite de pruebas
tools/            Paridad con PyTorch y Pillow, corpus, MNIST y OCR
benchmarks/       Mediciones de rendimiento
docs/             Guías, verificación y hoja de ruta
bin/              Binarios compilados
corpus/           Datos descargados (no se versionan)
release/          Modelos entrenados (no se versionan)
logs/             Logs de entrenamiento (no se versionan)
```

Las declaraciones viven en `include/` y los cuerpos en `src/`. Tres archivos son
la excepción y lo explican dentro: `serialization.h` y `autograd.h` por las
plantillas, y `parallel.h` porque su protocolo entre hilos no se entiende partido
en dos.

Los modelos entrenados se escriben siempre en `release/`, que no se versiona.
**Todavía no hay ningún [GitHub Release](https://github.com/iztaneo/NeuralSuite/releases)
publicado**: los resultados de [MODELOS.md](docs/MODELOS.md) son reproducibles
con los comandos y la procedencia que hay ahí, pero los pesos no se pueden
descargar aún.

---

## 📄 Licencia

El **código** se distribuye bajo la [Licencia Apache 2.0](LICENSE): uso libre,
incluido el comercial, con concesión expresa de patentes y la obligación de
dejar constancia de los cambios. Quién es el titular que firma cada cabecera
está en [AUTHORS](AUTHORS).

Los **modelos entrenados** que se adjuntan a un
[Release](https://github.com/iztaneo/NeuralSuite/releases) no son código y no
llevan licencia propia: son obra derivada de los datos con los que se
entrenaron, y la procedencia de cada uno está documentada en
[docs/MODELOS.md](docs/MODELOS.md) y en las notas del propio release.
