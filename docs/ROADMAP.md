# Hoja de ruta: Core Correctness & Architecture

Estado del plan que llevó a NeuralSuite de una colección de implementaciones a
un framework con sus operaciones verificadas. Cada punto cerrado enlaza el
commit que lo cerró.

El principio que ordena el plan: **cada operación debe ser correcta,
comprobable y reutilizable antes de añadir la siguiente arquitectura.**

Este documento dice **qué falta y en qué orden**. Lo que ya ocurrió —qué se
construyó, qué falló y qué se midió— está en
[history/DIARIO_FASES.md](history/DIARIO_FASES.md); el estado pieza a pieza, en
[ESTADO.md](ESTADO.md). Para **usar** el proyecto están las guías:
[LLM](GUIA_LLM.md), [difusión](GUIA_DIFUSION.md),
[autoencoder](GUIA_AUTOENCODER.md), [cómo se verifica](VERIFICACION.md) y las
[referencias bibliográficas](REFERENCIAS.md) de cada pieza.

## Las dos reglas

> **Prioridad de desarrollo: implementar primero capacidades reutilizables que
> habiliten modelos reales. No duplicar implementaciones existentes salvo que
> aporten corrección demostrable, rendimiento medible o una capacidad nueva del
> framework.**

> **Reference → Parity → Optimize → Benchmark → Integrate.**

El `Integrate` es el añadido reciente, y es el que marca la etapa actual. No
basta con que `Conv2D`, la atención, el autoencoder y la difusión funcionen cada
uno en su planeta: **la siguiente etapa consiste en que formen sistemas juntos.**

La primera regla nace de una medición, no de una intuición. Se construyeron
`LinearAutograd` y `EmbeddingAutograd` como parejas de verificación: ~330 líneas,
dos capas entre 2.7× y 20× más lentas que nadie usará para entrenar, y **cero
defectos encontrados**. El concepto quedó demostrado y ahí debe pararse. Migrar
`Conv2D`, `LSTM` y `MultiHeadAttention` al mismo esquema costaría otras ~600
líneas con el mismo retorno esperado, y **las tres ya tienen su pareja
`*Reference`**. El detalle, en [AUTOGRAD_CAPAS.md](AUTOGRAD_CAPAS.md).

Lo que sí necesita el autograd es dejar de ser una demostración y volverse
infraestructura: que permita **construir arquitecturas nuevas** sin escribir cada
backward a mano. Eso son dos operaciones concretas, no una fase entera.

## El foco: tres columnas y un puente

Una revisión externa, al cerrar la Fase 17, señaló el riesgo principal del
proyecto: **que la amplitud se vuelva su enemigo.** Ya hay GPT, LSTM, CNN, OCR,
GNN, GAN, autoencoder, difusión, tokenizador, serialización, decodificadores de
imagen y cargadores de datos, y es tentador seguir sumando. Pero a estas alturas
diez demos más aportan menos que conseguir que dos o tres familias funcionen
excepcionalmente bien. **NeuralSuite no debe convertirse en un museo de
arquitecturas.**

El trabajo nuevo se ordena alrededor de tres columnas:

```text
NeuralSuite
├── Lenguaje     Transformer / GPT
├── Visión       Conv / autoencoder / OCR
└── Generativo   difusión / difusión latente
```

y del puente que las une:

```text
contexto del Transformer  →  CrossAttention  →  difusión latente
```

**El filtro:** una pieza nueva entra si refuerza una columna o el puente. Si solo
añade otra arquitectura aislada, no entra, por interesante que sea.

Tres posturas que se derivan de esto y que conviene tener escritas:

- **El motor crece cuando una arquitectura real lo exige**, no por anticipado.
  La difusión ya dio el primer aviso sobre el backward manual; la difusión
  latente probablemente dará el segundo, y ese es el momento de decidir cuánto
  autograd hace falta.
- **El backend es una decisión aplazada, no olvidada.** CPU, `float32` y una sola
  máquina bastan para MNIST y para entender LDM. Cuando un experimento
  interesante tarde días, habrá que elegir entre seguir siendo un framework
  transparente de CPU o abrir una capa de backends.
- **No se llama *production-ready*.** CI, sanitizers, checkpoints sellados y
  paridad no lo convierten en infraestructura de producción: faltan estabilidad
  de API, versionado serio de modelos, perfiles de memoria, *fuzzing* más
  agresivo, backends acelerados y evidencia fuera de MNIST y de modelos
  pequeños. Lo que sí es, en una frase: *an independent C++17 neural-network
  framework for transparent training, inference and experimentation, with no
  external ML or linear-algebra runtime dependencies.*

**La meta de una 1.0** es demostrar que una misma infraestructura propia entrena
dos familias muy distintas —lenguaje y generación visual— con corrección
verificable: GPT moderno, difusión latente, paridad externa, datos reales,
entrenamiento reproducible, benchmarks y documentación sólida.

---

## Lo que falta

Nada de lo que queda está **mal**: son mejoras sobre código verificado, límites
del OCR con su causa medida, o construcción nueva.

**Ocho pendientes**, y ninguno pertenece a una fase abierta salvo el `dtype`:

### Lo siguiente: Fase 19 (LDM-2), difusión latente

La continuación natural. Todo lo que necesita ya existe y está verificado: la
`UNet2D`, los muestreadores, el autoencoder con su latente sellado y la
infraestructura de entrenamiento.

| | Qué |
| --- | --- |
| 1 | **Clasificador MNIST propio y FID** con sus características. Incluye la raíz de matrices simétricas (Jacobi) con paridad contra scipy |
| 2 | **Difundir sobre el latente** `8×8×4`, usando `(z − media) × escala` del checkpoint |
| 3 | **Comparar píxel contra latente** con el mismo presupuesto: tiempo, memoria y calidad |

**Criterio de salida:** generar dígitos reconocibles difundiendo en el latente, y
una comparación medida contra la difusión en píxeles. Lo que ya está medido es
que el camino cuesta **6.3× menos** por paso.

### Mejoras sobre lo verificado

| | Fase | Nota |
| --- | --- | --- |
| BPE propio | 08 | El tokenizador de caracteres gasta ~1 token por carácter; un BPE, ~0.25–0.3 |
| KV cache contiguo | 09 | Hoy es `vector<vector<float>>`, un vector por posición |
| Intrínsecos SIMD por arquitectura | 09 | Hoy el bucle interno lo autovectoriza el compilador; conviene medir antes de invertir |
| Estado del generador por hilo | 07 | El RNG es global. **Latente, no activo**: el entrenamiento se siembra por iteración |
| `dtype` | 13 | Lo único que deja abierta la Fase 13. Hoy todo es `float` y ningún caso de uso pide otra cosa |

### Límites conocidos del OCR, con la causa medida

El OCR **funciona**: 3.4 % de error de carácter sobre una página de libro que
nunca vio, frente al 0.1 % de Tesseract. Quedan tres límites, y de cada uno se
sabe por qué:

| | Medido | Qué haría falta |
| --- | --- | --- |
| Decidir qué banda es texto | El logotipo de Mitsubishi da dos bandas de basura | Ejemplos negativos en el generador. **Accionable** |
| No lee manuscrito | **160 %** de error, frente al 157 % del azar: ninguna señal | Un corpus de escritura a mano. **Bloqueado por datos** |
| No maneja varias columnas | Sin medir: la prueba sintética no valía | Un documento real a dos columnas. **Bloqueado por datos** |

### El recuento

Sale del propio documento y del diario, no de un número escrito a mano —que ya
divergió tres veces—:

```bash
grep -cE '^ *- \[x\]' docs/history/DIARIO_FASES.md docs/ROADMAP.md
```

---

## Estado por área

| Área | Estado |
| --- | --- |
| Corrección de gradientes | ✅ diferencias finitas, y paridad contra PyTorch en GPT, LSTM, BiLSTM, CRNN y todos los bloques de difusión y del autoencoder |
| Robustez de `Tensor` | ✅ formas validadas, sin estados inválidos, vistas sin copia |
| Testing | ✅ **60 pruebas**, validadas por mutación, con código de salida |
| Portabilidad | ✅ Linux (GCC/Clang), macOS y Windows en CI, Debug y Release, más ASan/UBSan |
| Serialización | ✅ formato NSF con versión, metadatos y checksum |
| API para terceros | ✅ `Parameter` y `Module` con registro automático |
| Autograd | ✅ motor, primitivas y LayerNorm compuesta; capas sin migrar |
| Rendimiento | ✅ **las tres capas con bucles escalares reformuladas**: Conv2D 55×, LSTM 16×, atención 17.6×. Falta SIMD explícito |
| Lectura de imagen | ✅ PNG, JPEG, BMP y Netpbm propios; 64 archivos byte a byte como Pillow |
| OCR | ✅ canal completo, **3.4%** de error sobre una página de libro (Tesseract 0.1%) |
| Estructura | ✅ interfaz en `include/`, implementación en `src/`, programas en `demos/`, `apps/` y `tests/` |
| Lenguaje | ✅ GPT entrenado en español, perplejidad **6.23** sobre un autor nunca visto; RoPE y KV-Cache deslizante |
| Difusión en píxeles | ✅ 80 000 iteraciones sobre MNIST, DDPM y DDIM, **genera dígitos nuevos** |
| Autoencoder latente | ✅ Fase 18: **33.2 dB**, y difundir en el latente cuesta **6.3× menos** que en píxeles |
| Difusión latente (LDM-2) | ⏳ Fase 19: generar sobre ese latente está **por hacer** |
| Reproducibilidad | ✅ reanudar da pesos **idénticos bit a bit**; checkpoints sellados y transaccionales |
| Documentación | ✅ guías de uso de los tres flujos y documento de verificación |

| Fase                       | Estado           | Fase              | Estado       |
| -------------------------- | ---------------- | ----------------- | ------------ |
| 00 Confianza y limpieza    | ✅               | 06 Serialización  | ✅           |
| 01 Corrección crítica      | ✅               | 07 Runtime y build| ✅ parcial   |
| 02 Tensor Core             | ✅               | 08 Tokenizador    | ✅ parcial   |
| 03 `Parameter` y `Module`  | ✅               | 09 Rendimiento    | ✅ parcial   |
| 04 Verificación matemática | ✅               | 10 Ecosistema     | ✅           |
| 05 Autograd                | ✅ parcial       | OCR (aparte)      | ✅ parcial   |
| 11 Estructura              | ✅               | 12 Deuda previa   | ✅           |
| 13 Cerrar el motor         | ⬜ falta `dtype` | 14 Vocabulario    | ✅           |
| 15 Transformer moderno     | ✅               | 16 Datos          | ✅           |
| 17 Difusión                | ✅               | 18 LDM-1          | ✅           |

El recuento sale del propio documento, no de un número escrito a mano —que ya
divergió tres veces—. El propio comando falló una cuarta: anclaba en `^- \[` y
no veía los pendientes indentados del OCR, así que informaba de 6 donde había 9.
Ahora ancla en `^ *- \[`:

```bash
echo "cerrados: $(grep -cE '^ *- \[x\]' docs/ROADMAP.md)  pendientes: $(grep -cE '^ *- \[ \]' docs/ROADMAP.md)"
```

---

## Rendimiento medido

Todas las cifras salen de `./benchmark` sobre un Apple M5, y todos los cambios
conservan el resultado **idéntico bit a bit**: la paridad con PyTorch devuelve
los mismos números después de cada uno.

| Cambio | Paso de entrenamiento | `MatMul` |
| --- | --- | --- |
| Punto de partida (un solo hilo, OpenMP ignorado en macOS) | 119 ms | 35 GF |
| Reparto entre hilos con `std::thread` | 52 ms | 130 GF |
| Reparto dinámico en vez de trozos iguales | 29 ms | 192 GF |
| Bloqueo de registros: cuatro filas de `C` a la vez | **27 ms** | **232 GF** |

Dos resultados negativos que conviene no repetir: reducir la memoria reservada
de 118 MB a 72 MB **no movió el reloj**, y el bloqueo de cache en el GEMM **no
aportó nada** —a veces empeoraba—. En ambos casos la intuición apuntaba a un
sitio y la medición a otro.

---

## Horizonte — sin casillas, deliberadamente

Lo que viene después de la Fase 19 es la **dirección declarada** del proyecto, no
un plan con casillas. Se escribe aquí para que el rumbo esté claro y para no
volver a discutirlo desde cero, pero no se trocea en tareas hasta que la fase
anterior esté cerrada y medida.

La razón es la que ya conoce este documento: **un plan de once fases inalcanzables
es otra lista que diverge**, y este proyecto ya arregló siete.

- **Compresión perceptual.** La GAN de `demos/demo_gan.cpp` dejaría de ser una
  demo aislada para convertirse en el discriminador por parches del autoencoder,
  como en el artículo de Rombach. Hoy el autoencoder entrena solo con pérdida
  cuadrática.
- **Condicionamiento texto→imagen.** `CrossAttention` ya existe y está verificada
  contra PyTorch, pero **ningún modelo la usa**: falta el puente entre el
  Transformer y la U-Net, y un corpus de pares texto-imagen que hoy no hay.
- **Muestreo guiado.** Sin condicionamiento no hay guía que aplicar; es el paso
  siguiente al anterior, no uno paralelo.

### Dos dependencias ocultas que hay que resolver antes de prometer nada

- **La pérdida perceptual necesita una VGG preentrenada**, que no se puede
  entrenar en CPU. O se importan pesos ajenos —y entonces deja de ser «solo
  NeuralSuite», que es la premisa del proyecto— o esa idea no se puede evaluar
  como está escrita. Hay que decidirlo antes, no al llegar.
  El mismo problema con **FID** se resolvió al abrir la Fase 18: se calculará con
  un clasificador MNIST propio, no con Inception.
- **BF16/FP16** en CPU sin AVX512-BF16 ni AMX no da ganancia, y convertir un
  framework que usa `float` en todas partes es un refactor grande. No es una
  opción de compilador.

---

## El diario de ingeniería

El detalle de las 19 fases ya cerradas —los defectos encontrados, las mediciones
y las decisiones— vive en **[history/DIARIO_FASES.md](history/DIARIO_FASES.md)**.
Se separó de aquí cuando este documento llegó a 2 176 líneas y dejó de servir
para saber qué falta, que es su trabajo.
