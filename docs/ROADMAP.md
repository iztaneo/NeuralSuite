# Hoja de ruta: Core Correctness & Architecture

Estado del plan que llevó a NeuralSuite de una colección de implementaciones a
un framework con sus operaciones verificadas. Cada punto cerrado enlaza el
commit que lo cerró.

El principio que ordena el plan: **cada operación debe ser correcta,
comprobable y reutilizable antes de añadir la siguiente arquitectura.**

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

---

## Lo que falta

Nada de lo que queda está **mal**: son mejoras sobre código verificado, límites
conocidos del OCR con su causa medida, o construcción nueva. Esa distinción es la
que ordena el trabajo, más que el número de fase.

### El orden, y por qué

| | Fase | Por qué va aquí |
| --- | --- | --- |
| **1. Deuda que bloquea entrenar** | 12 | Hoy **no se puede entrenar un segundo modelo sin destruir el primero**: `train_llm` no deja redirigir el vocabulario. Es barato y es un riesgo activo. Incluye el entrenamiento de referencia en español, que es la prueba de humo del canal completo. |
| **2. Cerrar el motor** | 13 | Sólo faltan `Concat` y `Backward(salida, gradiente)`. Dos operaciones pequeñas que desbloquean composición: sin `Concat` no hay skips de U-Net. Máximo desbloqueo por coste. |
| **3. Vocabulario compartido** | 14 | Seis piezas (`RMSNorm`, `SiLU`, `GroupNorm`, `Upsample2D`, `Downsample2D`, `CrossAttention`) que sirven a la vez al LLM y a la visión. Aquí el proyecto deja de acumular demos. |
| **4. Transformer moderno** | 15 | El examen principal, y está a mitad: ya hay KV-Cache, clipping, scheduler, checkpoint, paridad y corpus. |
| **5. Datos** | 16 | No existe ningún cargador de datasets. Bloquea todo lo visual. |
| **6. Difusión real** | 17 | Con MNIST como examen, no CIFAR: medido, la diferencia es 8 horas contra semanas. |

Y en paralelo, cuando convenga: BPE (Fase 08), SIMD y KV contiguo (Fase 09), y
los límites del OCR.

El recuento sale del propio documento con el comando que hay más abajo, no de un
número escrito a mano —que ya divergió tres veces—.

### Límites conocidos del OCR, con la causa medida

El OCR **funciona**: 3.4% de error de carácter sobre una página de libro que
nunca vio, frente al 0.1% de Tesseract. Lo que queda son tres límites, y de cada
uno se sabe por qué:

| | Medido | Qué haría falta |
| --- | --- | --- |
| No lee manuscrito | **160%** de error, frente al 157% del azar puro: ninguna señal | Un corpus de escritura a mano. No se puede generar renderizando tipografías |
| No distingue texto de dibujo | El logotipo de Mitsubishi da dos bandas de basura | Ejemplos negativos en el generador y un umbral de confianza |
| No maneja varias columnas | Sin medir: la prueba sintética no valía | Un documento real a dos columnas, y corte vertical antes del horizontal |

### Mejoras sobre lo verificado

| | Fase | Nota |
| --- | --- | --- |
| BPE propio | 08 | El tokenizador de bytes gasta ~1 token por carácter; un BPE gasta ~0.25–0.3. Secuencias 3–4× más cortas, y la atención es cuadrática en la longitud. (Medido: español 1.08 tokens/carácter, inglés 1.00 — los acentos son sólo el 8% del problema, no el motivo principal.) |
| Estado del generador por hilo | 07 | El RNG es global. **Verificado latente, no activo**: no hay `Dropout`, ninguna llamada aleatoria vive fuera de un constructor, y `RandomNormal`/`RandomUniform` son bucles secuenciales que ningún `ParallelFor` toca. Se vuelve defecto el día que se paralelice la generación de datos. |

### Rendimiento

Las tres capas con bucles profundos que no llamaban a `MatMul` —`Conv2D`, la
celda recurrente y la atención— ya están reformuladas, con **55×**, **16×** y
**17.6×** respectivamente. El barrido está cerrado; lo que queda es de otra
naturaleza.

| | Fase | Nota |
| --- | --- | --- |
| Intrínsecos SIMD por arquitectura | 09 | Hoy el bucle interno lo autovectoriza el compilador. |
| KV cache contiguo | 09 | Solo afecta a la generación token a token. Hoy es `vector<vector<float>>`. |
| RoPE | **15** | Movido: no es una mejora de rendimiento sino una pieza del transformer moderno. Requisitos en [FUTURE_PLAN_KVCACHE_ROPE.md](FUTURE_PLAN_KVCACHE_ROPE.md). |

---

## Estado por área

| Área | Estado |
| --- | --- |
| Corrección de gradientes | ✅ diferencias finitas, y paridad contra PyTorch en GPT, LSTM, BiLSTM y CRNN |
| Robustez de `Tensor` | ✅ formas validadas, sin estados inválidos, vistas sin copia |
| Testing | ✅ **34 pruebas**, validadas por mutación, con código de salida |
| Portabilidad | ✅ Linux (GCC/Clang), macOS y Windows en CI, Debug y Release, más ASan/UBSan |
| Serialización | ✅ formato NSF con versión, metadatos y checksum |
| API para terceros | ✅ `Parameter` y `Module` con registro automático |
| Autograd | ✅ motor, primitivas y LayerNorm compuesta; capas sin migrar |
| Rendimiento | ✅ **las tres capas con bucles escalares reformuladas**: Conv2D 55×, LSTM 16×, atención 17.6×. Falta SIMD explícito |
| Lectura de imagen | ✅ PNG, JPEG, BMP y Netpbm propios; 64 archivos byte a byte como Pillow |
| OCR | ✅ canal completo, **3.4%** de error sobre una página de libro (Tesseract 0.1%) |
| Estructura | ✅ interfaz en `include/`, implementación en `src/`, programas en `demos/`, `apps/` y `tests/` |

| Fase                       | Estado           | Fase              | Estado       |
| -------------------------- | ---------------- | ----------------- | ------------ |
| 00 Confianza y limpieza    | ✅               | 06 Serialización  | ✅           |
| 01 Corrección crítica      | ✅               | 07 Runtime y build| ✅ parcial   |
| 02 Tensor Core             | ✅               | 08 Tokenizador    | ✅ parcial   |
| 03 `Parameter` y `Module`  | ✅               | 09 Rendimiento    | ✅ parcial   |
| 04 Verificación matemática | ✅               | 10 Ecosistema     | ✅           |
| 05 Autograd                | ✅ parcial       | OCR (aparte)      | ✅ parcial   |
| 11 Estructura              | ✅               |                   |              |

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

## Fase 00 — Confianza y limpieza ✅

`a8a0629`, `ada1665`

- [x] **`ocr_cli` fabricaba su resultado.** Ignoraba `--image`, corría sobre
      ruido aleatorio y, si la red no predecía nada, escribía el literal
      `"MITSUBISHI MOTORS"` — justo el condicional que el comentario del archivo
      decía haber eliminado.
- [x] `demo_ocr` llamaba a un método inexistente: el archivo no compilaba.
- [x] `CRNNModel::DecodeWord` leía `Shape()[2]` sobre un tensor de rango 2.
- [x] Ejecutables, `libneuralsuite.a`, pesos y `resultado_*.txt` fuera del
      control de versiones.
- [x] `LICENSE` Apache-2.0 (las cabeceras la citaban sin que existiera).
- [x] Clase `Sequential` duplicada en dos cabeceras del mismo namespace.
- [x] `CMakeLists` pasa de 7 a 15 ejecutables: es la única fuente de verdad.
- [x] Convención `release/` para los artefactos de entrenamiento.

## Fase 01 — Corrección crítica del núcleo ✅

`3bcea54`

- [x] **`MultiHeadAttention` no sobrescribía `GetGradients()`.** Heredaba la
      lista vacía de `Layer`, de modo que cada `GPTBlock` exponía 12 parámetros
      frente a 8 gradientes. El optimizador aplicaba el gradiente de una capa a
      los pesos de otra y leía fuera de la lista corta.
- [x] **El weight tying perdía la contribución de la cabeza de salida.**
      `Embedding::Backward()` reinicia el acumulador después de que el modelo ya
      la había sumado. El modelo entrenaba y la pérdida bajaba, pero optimizaba
      una función distinta de la declarada.
- [x] `SGD` y `AdamW` validan que parámetros y gradientes casen en número y
      forma; antes el desajuste continuaba en silencio.
- [x] `Tensor` rechaza dimensiones negativas y desbordamiento de `size_t`.
- [x] `MatMul` exige rango 2 y dimensiones compatibles; `Embedding` acota el
      token; `MultiHeadAttention` exige `n_embd % n_head == 0`.
- [x] `CharTokenizer::Load` acota el vocabulario que lee del archivo.
- [x] Las pruebas dejan de usar `assert()`, que **desaparece bajo `NDEBUG`** y
      hacía que la suite pasara sin verificar nada en compilaciones Release.

## Fase 02 — Tensor Core ✅

`5d25002`, `44d0eb3`

- [x] **`Reshape` destruía datos.** Reasignaba y ponía a cero cuando cambiaba el
      número de elementos. Las 22 llamadas del repositorio querían esa
      reasignación, no la reinterpretación, así que la operación se separó:
      `Reshape` reinterpreta y falla si no cuadra; `Resize` reasigna.
- [x] `operator=` reservaba después de liberar: un `new` fallido dejaba un
      puntero colgante.
- [x] Comprobación de índices en compilaciones de depuración.
- [x] `NormalInit`, declarada sin definición, eliminada.
- [x] Almacenamiento compartido y `View()` sin copia. Medido sobre un paso de
      entrenamiento: 118 MB → 72,6 MB reservados. **El tiempo por paso no
      cambia**: el cuello de botella es el cómputo, no las reservas.
- [x] **Strides y `Transpose` como vista: medido y descartado.** Toda
      transposición del repositorio alimenta directamente un `MatMul`, así que
      se probó lo que una vista permitiría —leer el operando transpuesto sin
      copiarlo— implementando `MatMulNT` y `MatMulTN` y comparándolas contra
      transponer primero:

      | | transponer + MatMul | leer transpuesto |
      | --- | --- | --- |
      | `A · Bᵀ` 1024×512×512 | **210 GF** | 59 GF |
      | `Aᵀ · B` 1024×512×512 | 203 GF | 211 GF |

      Evitar la copia sale entre 3 y 4 veces más caro en el caso frecuente. Al
      leer `B` transpuesta, cada elemento de `C` pasa a ser un producto escalar:
      una reducción con dependencia en el bucle interno, que atasca el pipeline.
      La forma original no la tiene, porque cada iteración escribe una columna
      distinta.

      La conclusión invierte la premisa: **la copia no es un desperdicio, es lo
      que compra el patrón de acceso rápido**, y se paga sola con creces. Medido
      de extremo a extremo, el paso de entrenamiento empeoraba de 26.7 a 34.2 ms.
      Añadir strides al `Tensor` solo tendría sentido para operaciones que
      todavía no existen, y `Contiguous()` seria precisamente deshacer la vista
      para volver a este caso.

## Fase 03 — `Parameter` y `Module` ✅

`dc015ed`

- [x] **`Parameter` reúne valor y gradiente en un solo objeto**, creados juntos
      y con la misma forma. `GetParameters()` y `GetGradients()` dejan de ser
      dos métodos virtuales independientes y pasan a derivarse de la misma
      lista: ya no hay dos declaraciones que puedan discrepar, de modo que el
      defecto de la Fase 01 deja de ser representable en lugar de quedar
      atrapado por una guarda.
- [x] **`Module` con registro automático de submódulos.** Un `GPTBlock`
      enumeraba a mano sus cinco componentes en dos métodos que debían coincidir
      entre sí; ahora los declara una vez en el constructor y el recorrido del
      árbol es automático.
- [x] Los optimizadores aceptan una sola lista de `Parameter*`. Se conserva el
      constructor anterior, validado, para el código que aún pase dos listas.
- [x] Las doce demos y `train_llm` construyen el optimizador con la lista
      única: desaparece el patrón de armar dos vectores paralelos a mano.

## Fase 04 — Verificación matemática ✅

`3bcea54`, `349829f`, `45e9c15`, `1bbc847`

- [x] Gradient check de `GELU`, `MultiHeadAttention`, la matriz `wte`
      compartida, `LSTM`, `Conv2D` (con stride y padding activos), `LayerNorm`,
      `CrossEntropyLoss`, `MSELoss`, `MaxPool2D`, `ResidualBlock` y `GraphConv`.
- [x] **Cada comprobación validada por mutación**, introduciendo un defecto
      deliberado para confirmar que falla. No es ceremonia: reveló que la prueba
      de `GraphConv` usaba una adyacencia simétrica y por eso no detectaba que
      se omitiera la transposición.
- [x] **La capa `LSTM` se reimplementó entera.** No era una LSTM: su `Forward`
      calculaba `h = tanh(x[0] + h)` sin usar ninguno de sus cuatro parámetros,
      sin puertas y sin estado de celda; su `Backward` devolvía ceros.

### Sobre el paso de las diferencias finitas

No hay un valor universal, y elegirlo mal produce falsos positivos que parecen
defectos graves. Es un compromiso entre dos errores opuestos:

- **Paso pequeño**: `loss(w+ε) − loss(w−ε)` sufre cancelación en float32.
- **Paso grande**: domina el error de truncamiento, y en capas con ReLU o
  max-pooling la perturbación cruza el codo, midiendo el promedio de dos
  regímenes distintos.

Medido en este repositorio: en `MultiHeadAttention` el error baja de 0,48 a
3e-4 al **agrandar** el paso; en `GraphConv` sube de 1,8e-6 a 1,0 al agrandarlo.
Cada prueba lleva su calibración documentada en el código.

## Fase 05 — Autograd ✅ (motor y primitivas)

- [x] **Motor de diferenciación automática en modo inverso.** Cada operación
      registra cómo repartir el gradiente entre sus entradas, y `Backward()`
      recorre el grafo en orden topológico inverso. Un nodo que alimenta varios
      caminos acumula las contribuciones de todos.
- [x] Doce primitivas: `Add`, `Sub`, `Mul`, `MatMul`, `Sum`, `Mean`, `Exp`,
      `Log`, `Tanh`, `Relu`, `Reshape` y `Transpose`, cada una verificada por
      diferencias finitas y validada por mutación.
- [x] `demo_autograd` entrena XOR sin una sola derivada escrita a mano: la
      pérdida baja de 2.22 a 3e-05 y las cuatro predicciones son correctas.
- [x] **`Gather` y `Conv2DVar`**, las dos primitivas que faltaban. Eran las
      operaciones que el motor no sabía derivar solo, y por eso `Embedding` y
      `Conv2D` sólo existían como capas con backward escrito a mano. Cada una se
      verifica contra la capa equivalente, que ya tiene paridad con PyTorch. En
      `Gather` el caso que importa es el token repetido: su fila debe recibir la
      suma de todas las posiciones que lo usaron, y quedarse con la última en vez
      de sumarlas pone la prueba en rojo.
- [x] **`LinearAutograd`**, primera capa migrada. Misma interfaz que `Linear` y
      mismos parámetros en el mismo orden, pero su `Backward` no lo escribió
      nadie: sale de encadenar `MatMulVar` y `Add`. **Convive con `Linear` en vez
      de sustituirla.** `Linear` sigue siendo la de entrenar —el grafo reserva los
      intermedios de cada pasada—; la del grafo es la que dice si `Linear`
      acierta, desde un camino que no puede repetir la clase de error de los dos
      defectos P0, porque no aplica ninguna fórmula escrita a mano.
- [x] **`EmbeddingAutograd`**, segunda capa migrada. Toda la capa es una llamada
      a `Gather`. Resuelve sin codigo propio el caso que usa el GPT —el embedding
      de posicion se calcula con forma `[1, T]` y su gradiente llega `[B, T, D]`,
      asi que cada posicion suma las `B` contribuciones—: le sale del
      broadcasting. `Embedding` lo consigue con un `% num_cached` que parece
      defensivo y en realidad es carga estructural, cosa que ahora una prueba
      unitaria fija.
- [x] **Decidido NO migrar el resto de capas.** El concepto quedó demostrado con
      `Linear` y `Embedding`, y lo que se midió fue: ~330 líneas, dos capas entre
      2.7× y 20× más lentas que nadie usará para entrenar, y **cero defectos
      encontrados**. `Conv2D`, `LSTM` y `MultiHeadAttention` costarían otras ~600
      líneas con el mismo retorno esperado, y **las tres ya tienen su pareja
      `*Reference`**, así que el oráculo adicional aporta todavía menos que en las
      dos que no la tenían. Es la primera de [las dos reglas](#las-dos-reglas): no
      duplicar salvo que aporte corrección demostrable, rendimiento medible o una
      capacidad nueva. El esfuerzo del autograd se traslada a la Fase 13, que sí
      habilita arquitecturas nuevas.

  **Antes de seguir migrando, leer [AUTOGRAD_CAPAS.md](AUTOGRAD_CAPAS.md)**, que
  recoge lo medido: las versiones del grafo son 2.7x y 20x mas lentas, no han
  encontrado ni un defecto, y las tres capas que quedan ya tienen su pareja de
  referencia, asi que el oraculo adicional aporta menos que en `Linear` y
  `Embedding`. Ese documento tambien deja las cuatro mejoras concretas por si
  se retoma.

  La duplicación sólo es segura porque hay una prueba que enfrenta a cada
  pareja. Es la diferencia medida entre los tres pares que el proyecto ya
  mantiene —`Conv2DReference`, `LSTMReference`, `MultiHeadAttentionReference`,
  que no han divergido nunca— y las seis listas hechas a mano que nadie
  comparaba, que divergieron todas.
- [x] **Broadcasting** en las operaciones elemento a elemento, con la regla
      habitual de alinear por la derecha. Un eje que se repite en el forward
      recibe en el backward la suma de todas las posiciones que lo usaron. Sin
      esto un sesgo `[D]` no podía sumarse a un lote `[N, D]`, y el propio
      `demo_autograd` tuvo que declararlo con la forma del lote.
- [x] **Softmax** como primitiva sobre el último eje. Se implementa directamente
      y no por composición porque la estabilidad numérica exige restar el máximo
      de cada fila, y expresarlo con primitivas obligaría a derivar también por
      ese máximo.
- [x] **LayerNorm compuesta de primitivas, sin backward propio.** Es la razón de
      tener autograd: la versión escrita a mano necesita una fórmula de tres
      términos para `dx`, y omitir uno produce un gradiente equivocado que no da
      síntoma. Aquí se declara el cálculo hacia delante y la derivada sale sola;
      coincide con la implementación manual a 1e-4 y su gradiente con
      diferencias finitas a 4.7e-4.
- [x] Reducciones por eje (`SumLastAxis`, `MeanLastAxis`), `Div`, `Sqrt` y
      `AddScalar`, que son las que permiten componer normalizaciones.
- [x] `gather` y convolución como primitivas, verificadas contra `Embedding`
      y `Conv2D` como oráculo (detalle más arriba, en la Fase 05).

Reduce la superficie de error: cada capa implementaba su backward a mano, que es
exactamente donde aparecieron los dos defectos P0.

## Fase 06 — Serialización ✅

`051b952`, `0775e18`

- [x] **Formato NSF**: número mágico, versión, metadatos de la arquitectura,
      tensores con nombre y forma, y suma de comprobación.
- [x] `Module::NamedParameters()` da a cada peso su ruta en el árbol
      (`blocks.0.attn.c_attn.weight`), de modo que el archivo es
      autodescriptivo en lugar de una secuencia numerada.
- [x] `GPTModel`, `CRNNModel` y `Sequential` lo usan.

Antes se volcaban floats crudos sin cabecera: un checkpoint de otra
arquitectura se cargaba sin dar error y el modelo se quedaba con datos sin
sentido. Ahora la carga rechaza, indicando el motivo, un archivo de otra
arquitectura, truncado, con un byte alterado, o sin cabecera. Los pesos
guardados con el formato anterior no se pueden cargar y el mensaje lo dice.

## Fase 07 — Runtime y build portable ✅ (parcial)

`cb0c07d`, `49ef343`

- [x] **El build de CMake no funcionaba en macOS.** `find_package(OpenMP
      REQUIRED)` falla con AppleClang, así que la instrucción principal del
      README nunca se había ejecutado en una de las tres plataformas que
      anuncia. OpenMP pasa a ser opcional.
- [x] `-march=native` y `-ffast-math` dejan de aplicarse siempre. La primera
      producía binarios que podían no arrancar en otra CPU; la segunda relaja
      IEEE-754 justo donde se verifican gradientes.
- [x] **El `Makefile` arrastraba los mismos defectos, sin corregir.** Al quitar
      OpenMP quedó `CXXFLAGS += -Xpreprocessor -fopenmp 2>/dev/null || true`, y
      make pasaba esos tres tokens al compilador como ficheros: en macOS no
      producía ningún objeto. Seguía forzando `-march=native` y `-ffast-math`,
      así que este build y el de CI no calculaban lo mismo. Y mantenía a mano
      tres listas de programas que ya habían divergido: `demo_autograd` no se
      compilaba y `clean` borraba cuatro de los quince binarios. Ahora los
      programas salen de un `wildcard` y hay una sola regla.
- [x] **Windows nunca había compilado**: `#pragma omp simd` requiere un flag
      experimental en MSVC, y su OpenMP exige índice de bucle con signo.
- [x] `ManualSeed()` fija la semilla del generador. Antes reproducir una
      inicialización dependía de que nadie hubiera consumido números antes:
      construir una capa de más desplazaba todo lo que viniera después.
- [ ] Estado del generador por hilo. Sigue siendo compartido y no es seguro
      usarlo desde varios hilos a la vez.

## Fase 08 — Tokenizador ✅ (parcial)

- [x] **Token `<UNK>` explícito** en el índice 0. Antes un símbolo fuera del
      vocabulario se codificaba como token 0, que era un carácter válido: con el
      vocabulario `{a, b, c}`, codificar y decodificar `"axc"` devolvía `"aac"`,
      convirtiendo la `x` desconocida en una `a`. Ahora devuelve `"a?c"`, y
      `CountUnknown()` permite medir cuánto del corpus queda fuera.
- [x] **`ByteTokenizer`** con vocabulario fijo de 256 símbolos. Por construcción
      no puede encontrarse un símbolo desconocido, y no necesita reentrenarse
      para otro idioma: el mismo tokenizador reproduce japonés o emoji sin
      haberlos visto. Mantiene la dependencia cero, porque tratar bytes no
      requiere ninguna biblioteca Unicode.
- [ ] BPE propio, para que las secuencias no crezcan tanto: en el tokenizador
      de bytes un carácter no ASCII ocupa varios tokens.

## Fase 09 — Rendimiento ✅ (parcial)

- [x] **Perfilado antes de optimizar.** `MatMul` resultó ser el 80% del tiempo
      de un paso de entrenamiento; el resto se reparte entre los bucles de
      atención (que a su vez llaman a `MatMul`), GELU y `Transpose`. LayerNorm y
      softmax juntos no llegan al 1%.
- [x] **Paralelismo con `std::thread`** en `include/parallel.h`, sustituyendo a
      OpenMP. Contrastado contra el propio OpenMP —instalándolo aparte y
      repartiendo el mismo bucle de las dos formas— para descartar que la
      implementación casera dejara rendimiento sobre la mesa. En un Apple M5
      están a la par (media ~1.02x); en un runner Linux de 4 vCPU, sobre una
      base en serie de 3.1 GFLOP/s, el reparto propio da 1.76x y OpenMP 2.00x
      —un 12% de diferencia, no el 2x que sugerían las primeras cifras, tomadas
      sin medir el caso de un solo hilo—. Ese margen no compensa mantener dos
      rutas: duplicaría la superficie a probar, el OpenMP 2.0 de MSVC obliga a
      índices con signo, y sobre todo OpenMP degrada en silencio cuando falta. El motivo no era preferencia: con AppleClang los `pragma omp` se
      ignoran, de modo que en macOS todo corría en un solo hilo y `MatMul` usaba
      una cuarta parte de la máquina. Medido sobre un Apple M5 de cuatro núcleos
      de rendimiento: `MatMul` pasa de 35 a 232 GFLOP/s, y el paso de
      entrenamiento de 119 ms a unos 27 ms.
- [x] **Reparto dinámico en lugar de trozos iguales.** Con un trozo por hilo, el
      tiempo lo marca el más lento, y los núcleos rara vez son iguales: un
      Apple M5 mezcla núcleos de rendimiento y de eficiencia, y en un servidor
      dos hilos pueden compartir el mismo núcleo físico. Repartir bloques que
      cada hilo toma al desocuparse sube la aceleración de 3.7x a 5.3x y el paso
      de entrenamiento de 37 ms a 29 ms. Es justo el margen que separaba al
      reparto propio de OpenMP. El resultado es idéntico bit a bit,
      porque cada hilo escribe filas disjuntas y no hay reducción que altere el
      orden de las sumas.
- [x] **Bloqueo de registros en el GEMM: cuatro filas de C a la vez.** El bucle
      leía una fila entera de `B` para producir una sola fila de `C`, de modo
      que cada elemento de `B` viajaba de memoria a registro para una única
      multiplicación. Medido en un solo hilo: de 35 a 44 GFLOP/s.
      Cuatro y no más — con seis u ocho acumuladores el compilador se queda sin
      registros vectoriales y cae a 41 y 33.
      **El bloqueo de cache no aportó nada** (a veces empeoraba): las matrices
      de estos tamaños ya caben, y lo que faltaba era reutilizar los datos ya
      cargados, no traerlos mejor.
- [x] **Repartir entre hilos las operaciones elementales y el pooling.** No
      había nada que reformular: `ReLU`, `Sigmoid`, `Tanh`, las tres
      elementales y `MaxPool2D` simplemente no usaban el pool de hilos, por
      omisión y no por decisión —`GELU` y el softmax sí lo hacían desde el
      principio—. Una vez que la convolución y la celda recurrente dejaron de
      dominar, eran **15.2 ms de un paso de 28.2, el 53.8%**. Ahora **4.6 ms**,
      y el paso completo baja a **19.5 ms**: 486 imágenes por segundo frente a
      las 6.6 del punto de partida, **62×**.

      El detalle que importa está en el backward de `MaxPool2D`: el reparto va
      por plano y no por posición de salida. Dos ventanas solapadas —cuando el
      paso es menor que la ventana— pueden compartir máximo, de modo que dos
      posiciones de salida suman sobre la misma de entrada; repartir por
      posición sería una carrera. El test 30 lo comprueba con ventana 3 y paso
      2, y esa mutación exacta lo pone en rojo.
- [x] **Kernel optimizado de `MultiHeadAttention`**, el tercero y último del
      mismo patrón, con el literal conservado como `MultiHeadAttentionReference`.
      `Q·Kᵀ` y `P·V` son multiplicaciones de matrices escritas como bucles que
      recorrían cada elemento, a 1.3 GFLOP/s y en un solo hilo. Hizo falta
      reordenar memoria: `c_attn` entrega Q, K y V entrelazados en una fila de
      `3·n_embd` y cada cabeza ocupa un tramo, así que hay que extraerlas como
      matrices contiguas — el mismo trabajo que `im2col`. Se calcula el cuadrado
      completo de puntuaciones, incluida la mitad que la máscara descarta: el
      doble de operaciones que el bucle triangular, y aun así mucho más rápido.

      Medido sobre un paso del GPT (lote 16, 4 capas, n_embd 128):

      | Contexto | Antes | Después | |
      | --- | --- | --- | --- |
      | 64 | 483 ms | **126 ms** | 3.8× |
      | 128 | 1669 ms | **258 ms** | 6.5× |
      | 256 | 6034 ms | **556 ms** | 10.9× |
      | 512 | 23410 ms | **1331 ms** | **17.6×** |

      Lo que **no** cambia es que el coste crezca con el cuadrado del contexto:
      esa es la definición del mecanismo. Lo que cambia es la constante, y con
      ella el crecimiento observado pasa de 48.5× a **10.6×** al ir de contexto
      64 a 512 — porque ahora la parte cuadrática ya no domina desde el primer
      momento. Un entrenamiento de 1000 iteraciones con contexto 512 baja de
      6 h 30 a 22 minutos.

      El reparto va por pareja (muestra, cabeza) y cada cabeza escribe en un
      tramo distinto de `dqkv`, así que no hay reducción y el resultado es
      idéntico con uno o con diez hilos. El test 33 lo exige, junto con seis
      configuraciones contra la referencia; cinco mutaciones lo ponen en rojo.
- [x] **Kernel optimizado del LSTM**, con el literal conservado como
      `LSTMReference`. La preactivación de las puertas es
      `x_t·W_ihᵀ + h_{t-1}·W_hhᵀ`, dos multiplicaciones de matrices que estaban
      escritas como productos escalares unidad por unidad: iba a 1.27 GFLOP/s
      con `MatMul` a 232. La proyección de la entrada no depende del estado
      anterior, así que los 32 pasos salen en una sola multiplicación; la parte
      recurrente sigue siendo secuencial. En el backward la asimetría se
      acentúa: solo `dpre` va paso a paso, y apilarlos convierte `dW_ih`,
      `dW_hh` y `dx` en tres multiplicaciones grandes —el truco está en que
      `dW_hh` suma sobre pasos y lote a la vez—. **El BiLSTM baja de 159 a 9.8
      ms (16.2×) y el paso de entrenamiento de 183 a 28.2 ms.** Frente al
      original de 1213 ms son **43×**, y 306 imágenes por segundo en vez de 6.6.
      El cuello pasa a ser `ReLU` y `MaxPool2D`, con el 53.8%.
- [x] **Kernel optimizado de Conv2D**, con el literal conservado como
      `Conv2DReference` en el mismo archivo. Medido antes: las tres
      convoluciones eran el 85.7% de un paso de entrenamiento del CRNN y
      corrían en un solo hilo teniendo diez. Reformuladas como `im2col` +
      `MatMul` —que ya estaba paralelizado y con bloqueo de registros— van
      **54.8× más rápido** ellas solas y el paso completo **6.2×**: de 1213 ms
      a 195 ms, de 6.6 a 41 imágenes por segundo. La convolución pasa del 85.7%
      al 8.9% del paso.
- [ ] Intrínsecos SIMD por arquitectura. Hoy el bucle interno lo autovectoriza
      el compilador.
- [x] **KV-Cache**, con `--no_cache` para contrastarlo. Medido con 4 capas y 64
      tokens generados: **9.6× más rápido** (0.36 ms/token frente a 3.50), y las
      dos rutas generan **exactamente la misma secuencia**, o sea que acelera sin
      cambiar el resultado.
- [ ] KV cache contiguo (hoy `vector<vector<float>>`: un vector por posición, con
      su reserva propia).
- [x] **Retirado el esbozo de RoPE.** `ApplyRoPE()` no lo llamaba ningún
      forward: una función así sugiere una capacidad que el modelo no tiene, y es
      peor que su ausencia porque nadie sabe que no hace nada. La posición sigue
      llegando por embeddings aprendidos.
  **RoPE se trasladó a la Fase 15**, donde le corresponde: no es una mejora de
  rendimiento sino una pieza del transformer moderno. Su casilla vive allí, y no
  aquí, para que no cuente dos veces.

  Merece la pena dejar escrito por qué estuvo invisible tanto tiempo. La casilla
  decía «RoPE decidido» y estaba marcada como hecha, pero lo decidido fue
  *retirar el esbozo muerto*, no que la técnica estuviera resuelta. Una decisión
  marcada como logro esconde el trabajo que queda.

## Fase 10 — Ecosistema ✅

`cb0c07d`, `49ef343`

- [x] CI en Linux (GCC y Clang), macOS y Windows, en Debug y Release; suite
      numérica, demos, entrenamiento de extremo a extremo, paridad con PyTorch y
      sanitizers.
- [x] Comparación contra la implementación de referencia en PyTorch.
- [x] **Benchmarks separados** en `benchmarks/benchmark.cpp`. Responden "cuánto
      tarda", que es otra pregunta y con otro criterio: no hay nada que pase o
      falle, solo números que comparar entre versiones. Existen porque las
      mediciones que guiaron la paralelización se hicieron a mano y no quedaban
      en el repositorio, de modo que nadie podía reproducirlas ni notar una
      regresión. Miden `MatMul` por forma, el escalado con hilos y el paso de
      entrenamiento en tokens/s, y comprueban de paso que el reparto no altera
      el resultado.
- [x] `SoftmaxForward` opera sobre el último eje de cualquier rango. Antes
      exigía rango 2 y leía `Shape()[1]` directamente, de modo que un tensor de
      rango 3 se normalizaba sobre un eje que no le correspondía.
- [x] **Grupos de parámetros en AdamW.** El decay se declara por grupo en vez
      de inferirse del rango del tensor: la heurística anterior acertaba solo
      porque la convención habitual coincide con ella, y un parámetro 2D que no
      debiera decaer recibía el trato equivocado sin que nada lo indicara.

## Pendiente aparte — OCR ⬜

El único punto del plan donde el repositorio anuncia algo que no hace: `README`,
`demo_ocr` y `ocr_cli` hablan de OCR, pero `CRNNModel` no lee imágenes y su
arquitectura no es la que declara. La referencia es
`LLMRasec/src/ocr.py`: `Conv×3 → BiLSTM → Linear`.

- [x] **`BiLSTM`.** Dos celdas independientes, una por sentido, con la salida
      concatenada `[T, B, 2H]`. No reimplementa la recurrencia: invierte el eje
      temporal y reutiliza la `LSTM` ya verificada, porque duplicar el BPTT es
      justo donde es fácil equivocarse. Verificada por tres vías distintas —
      gradient check, una prueba de direccionalidad que mide *de qué depende*
      cada salida, y paridad contra `nn.LSTM(bidirectional=True)`, que sale
      dentro de 5.3e-06.
- [x] **Arquitectura real.** `1→16→32→64` con pools `2,2` / `2,2` / `8,1`,
      luego `BiLSTM(64, hidden)` y `Linear(2·hidden, clases)`. La red devuelve
      ahora una predicción por cada cuatro columnas en vez de una por imagen:
      la diferencia entre leer una palabra y clasificar un carácter suelto.
      `MaxPool2D` admite ventana rectangular, que es lo que hacía falta para
      colapsar el alto sin tocar el ancho. Tres defectos que salieron por el
      camino: la versión anterior compartía una única `Activation` entre dos
      puntos de la red —y `Activation` guarda su entrada para el backward, así
      que la segunda pisada borraba la caché de la primera—; `SynthTextGenerator`
      devolvía ruido gaussiano con etiquetas `i % 4`, sobre el que ninguna red
      podía acertar más que por azar; y las dos conversiones de disposición
      estaban escritas como funciones distintas cuando calculaban lo mismo.
      `demo_ocr` transcribe las 4 líneas del lote sin errores.
- [x] **Paridad del CRNN completo** contra `LLMRasec/src/ocr.py`. Logits en
      2.2e-05, `dx` hacia la imagen en 2.5e-04, y 15 de los 16 gradientes por
      debajo de 3e-04. El que se sale —`conv2.weight`, en 2.3e-03— resultó ser
      redondeo: el exportador calcula además el modelo en float64 y ambas
      implementaciones se apartan de él por igual (6.1e-03 PyTorch, 8.3e-03
      C++), porque el gradiente de una convolución suma miles de términos en un
      orden que no coincide entre las dos. Ejercita de paso el camino
      convolucional de extremo a extremo, que hasta ahora solo tenía gradient
      check — y ya se vio con `GraphConv` que eso puede no comprobar nada.
- [x] **Lector de imagen.** `include/image/` decodifica PNG, BMP y Netpbm sin
      ninguna dependencia externa. El PNG obligó a escribir *inflate* completo
      (RFC 1951 y 1950, con Huffman dinámico y comprobación del Adler-32), y
      admite profundidades de 1 a 16 bits, los cinco tipos de color, paleta con
      `tRNS` y entrelazado Adam7. Los 46 archivos del banco se decodifican byte
      a byte igual que Pillow, incluidos los tres PNG que ya estaban en el
      repositorio. `ocr_cli --image` ya lee de verdad el archivo que se le pasa.
- [x] **Tubería de entrenamiento sobre texto real.** `tools/ocr/` genera un
      corpus con tipografías del sistema y `train_ocr` lo entrena leyendo las
      imágenes con el decodificador propio: entrenar no necesita Python.
      Comprobado que la maquinaria es correcta con la prueba que lo decide —
      sobreajustar ocho imágenes: la pérdida baja de 4.14 a 0.13 en 400 pasos.
      Sin CTC: como el corpus lo dibujamos nosotros, se conoce en qué columnas
      cae cada letra y basta `CrossEntropyLoss`.
- [x] **Entrenamiento largo: hecho, y con un límite encontrado.** La pérdida
      seguía bajando en la época 30, así que se entrenó hasta 60. **Todas las
      cifras de validación mejoraron** —pérdida 0.150 → 0.113, acierto por
      palabra 37.7% → 41.6%, error de carácter 0.122 → 0.112— y sobre imágenes
      reales no ganó nada:

      | | 30 épocas | 60 épocas |
      | --- | --- | --- |
      | Ilíada | 3.5% | 3.4% |
      | Mitsubishi | **0.0%** | **12.5%** |

      `MITSUBISHI` pasó a leerse `MIlTSUBIlSHI`. Es la misma trampa que apareció
      al cambiar la forma de los datos, y más afilada: **la validación sale del
      mismo generador que el entrenamiento, así que mejora con él**. Un modelo
      que se ajusta mejor a su propio corpus no lee mejor una página que nunca
      vio.

      Los pesos publicados son los de 30 épocas. La conclusión es negativa y por
      eso conviene anotarla: el camino para bajar del 3.4% no es entrenar más.
- [x] **Leer una página, no un renglón.** `ocr_cli --renglones` corta la página
      y transcribe cada línea. Estado actual, medido con
      `tools/ocr/evaluar.py`:

      | | NeuralSuite | Tesseract |
      | --- | --- | --- |
      | Página de libro (Ilíada) | **3.4%** | 0.1% |
      | Logotipo (`MITSUBISHI`, `MOTORS`) | **0.0%** | 0.0% |

      Las tres piezas de las que dependía, y cómo quedaron:
      La vara de medir es Tesseract, que sobre la página original da **0.1%**
      frente a nuestro **54%**. Esa es la distancia real a un OCR maduro, y
      conviene tenerla delante.

      Cuánto se aleja cada caso de lo que el modelo sabe leer, medido en error
      de carácter: **2.3%** en la validación sintética, **54%** en la página de
      la Ilíada —impresa, pero en serif pequeña y con renglones cinco veces más
      largos— y **160% en un abecedario manuscrito, frente al 157% que da
      generar caracteres al azar**. Sobre manuscrito no hay señal ninguna.

      De la Ilíada salió además una corrección: se atribuía el fallo a los
      acentos y a los espacios, y son 3.5 puntos de 53.6. El grueso es el
      modelo leyendo mal un régimen que no vio.

      Nota metodológica que casi se pasa por alto: la transcripción de
      referencia de esa página la escribió el asistente **leyendo la imagen**,
      con la misma facultad que puede rellenar palabras plausibles. Medir un OCR
      contra algo que alucina no es medir. Se verificó contra Tesseract: 16 de
      las 18 líneas coinciden carácter a carácter, y las dos discrepancias son
      `LIBRO I` frente a `LIBRO 1` y `Leto;` frente a `Leto:` — dos caracteres
      de 853, el 0.23%. La referencia se sostiene, pero el orden correcto era
      comprobarlo antes de publicar el número.
      - [x] **Cortador de renglones** (`include/image/renglones.h`, `ocr_cli
        --renglones`). Proyección horizontal con umbral de Otsu, que sale del
        histograma en vez de ser una constante. Encuentra los 18 renglones de la
        Ilíada y separa el logotipo de Mitsubishi de sus dos palabras. El
        recorte no es al ras: se calcula el margen que deja la tinta ocupando el
        72% del alto, que es la proporción medida sobre 800 imágenes del corpus
        — al ras, cada letra abarca más pasos de los que el modelo vio y aparecen
        caracteres insertados. `MITSUBISHI` pasó de `MIlT5SUBlISHI` a
        `MITPSUBISHI` y `MOTORS` de `M0OTO0O0RS` a `MOT0RS` solo con eso.
      - [x] **Forma de los datos** (corpus `ocr_v2`). El modelo se entrenaba con
        palabras sueltas de ≤10 caracteres en ocho tipografías sans-serif, y se
        le pedía leer renglones de libro de ~50 en serif pequeña. Cuatro
        cambios: 154 tipografías en vez de 8, renglones de varias palabras,
        512 px de ancho, y degradación (reducir y reampliar, desenfoque, ruido).

        El cambio de fondo fue la **clase de blanco**: la clase 62 hacía de
        espacio real y de marca «aquí no hay letra» a la vez, y el decodificado
        la descartaba siempre. Por construcción, el modelo no podía leer una
        línea de varias palabras. **53% → 7.6%.**
      - [x] **Vocabulario con acentos y puntuación** (corpus `ocr_v3`). De 63 a
        91 símbolos más el blanco. **7.6% → 3.4%.**

        Obligó a cambiar la indexación: el vocabulario se leía como
        `std::vector<char>` —un byte por clase— y en UTF-8 `á` ocupa dos bytes y
        `—` tres, así que la clase 26 habría dejado de ser una letra. Ahora es
        `std::vector<std::string>` con un partidor de UTF-8.

        El *tofu* reapareció en versión sutil: 31 de las 154 tipografías dibujan
        cajas vacías para los acentos —coreanas, bengalíes, de símbolos, con las
        latinas básicas pero sin `ñ`—. El primer detector tampoco valió, porque
        comparaba contra el glifo U+FFFD y cada fuente sustituye con lo que le
        parece. El criterio bueno es el de siempre: si la fuente tiene los
        caracteres, cada uno se dibuja distinto. Quedan 122 verificadas.
      - [x] **Herramienta de medida** (`tools/ocr/evaluar.py`), que comprueba
        **antes de medir**. Existe por un fallo: una medición ejecutó `ocr_cli`
        con `2>/dev/null`, tiró el aviso de que los pesos no habían cargado, y
        midió una red sin entrenar como si fuera un resultado —98% de error—.
        La conclusión que casi se reporta era que los acentos habían empeorado
        el modelo doce veces. Ahora verifica código de salida y avisos de carga,
        y aborta con código 1 en vez de dar una cifra. Pone al lado la de
        Tesseract, que era otro punto del backlog.
      - [ ] **Decidir qué banda es texto.** Es lo único que queda de esta parte.
        El cortador entrega también las bandas del dibujo de un logotipo, y el
        reconocedor devuelve basura sobre ellas porque nunca vio ejemplos
        negativos. Filtrar por altura funcionaría en ese logotipo y sería
        tropicalizar; la salida general es que lo decida el propio reconocedor
        por su confianza, lo que exige **bandas sin texto en el generador**
        —manchas, líneas, fragmentos gráficos— etiquetadas como vacías.

        Mientras tanto, `evaluar.py` marca ese caso con `solo_texto` para no
        mezclar el fallo de detección con el de reconocimiento en una sola
        cifra.
      - [ ] **Columnas.** Bloqueado: hace falta un documento real a dos
        columnas. La prueba sintética que se hizo comprimía la página a la mitad
        de ancho, deformando las letras, así que medía dos cosas a la vez.
      - [ ] **Manuscrito.** Bloqueado por los datos, igual que las columnas, y
        aparecía en la tabla de límites sin casilla propia. **160%** de error
        frente al **157%** de generar caracteres al azar: no hay señal ninguna.
        El corpus se genera renderizando tipografías, y eso no produce escritura
        a mano. No es una mejora del modelo: es una fuente de datos que hoy no
        existe.

- [x] **JPEG**, secuencial de línea base y progresivo. Huffman, cuantización,
      transformada inversa e interpolación de crominancia. Es el único formato
      cuya salida **no está especificada bit a bit**: la norma fija requisitos
      de precisión para la transformada (T.83), no un resultado exacto, así que
      la comparación con libjpeg necesita un criterio estadístico y la
      transformada se verifica aparte contra su definición matemática. El modo
      aritmético, el sin pérdida y el de 12 bits se rechazan diciendo cuál es.

---

## Fase 11 — Estructura del repositorio ✅

`v0.9.0` marca el estado anterior a este cambio, por si hay que volver.

- [x] **Interfaz e implementación separadas.** La biblioteca era de solo
      cabeceras: 34 archivos y 8356 líneas en `include/`, con 676 en `src/`.
      Para saber qué ofrecía el `LSTM` había que atravesar 665 líneas cuando su
      interfaz son 27. Los cuerpos pasan a `src/`, con las mismas subcarpetas.

      | | Antes | Cabecera | Implementación |
      | --- | --- | --- | --- |
      | `jpeg.h` | 780 | 218 | 585 |
      | `lstm.h` | 664 | 259 | 430 |
      | `attention.h` | 592 | 199 | 422 |

      En total, de 8356 a 4826 líneas de cabecera y de 676 a 4722 en `src/`,
      repartidas en 25 archivos. Compilar todo baja de 7.9 a 5.3 segundos.

      **Qué no se mueve, y por qué.** Los constructores se quedan: su lista de
      inicialización dice en qué orden se construye la clase, que es diseño. Los
      accesores de una línea, también: son interfaz. Y tres archivos enteros —
      `serialization.h` y `autograd.h` por las plantillas, y `parallel.h` porque
      lo que hay que entender ahí es el protocolo entre hilos y no se parte en
      dos archivos sin perderlo. El motivo va escrito dentro de cada uno.

      La transformación destapó una dependencia oculta: `jpeg.h` y `enderezar.h`
      usaban `kPi` sin incluir `tensor.h`, y compilaban porque otro archivo lo
      arrastraba antes.
- [x] **Programas fuera de la raíz**, que tenía 17 `.cpp` mezclados. Van a
      `demos/`, `apps/` y `tests/`, y los binarios a `bin/`.

**Seis listas escritas a mano se quedaron cortas durante este trabajo**, y
conviene tenerlas juntas porque es el error que más veces se ha repetido en el
proyecto:

| Dónde | Qué faltaba | Consecuencia |
| --- | --- | --- |
| `.gitignore` | `demo_autograd` | Un binario de 100 KB versionado |
| `.gitignore` | `train_ocr` | Otro binario versionado |
| `Makefile`, objetivos | `demo_autograd` | Solo se construía con CMake |
| `Makefile`, fuentes | `src/image/`, `src/layers/` | No enlazaba |
| `run_parity.sh` | `src/layers/` | Símbolos sin definir |
| `CMakeLists.txt` | `train_ocr` | **Días sin construirse con CMake, ni en el CI** |

Todas pasan a derivarse de un patrón. CMake desaconseja `GLOB` porque no se
entera de los archivos nuevos sin reconfigurar; aquí pesa más que nadie se
acuerde de tocar la lista.

---

## Fase 12 — Deuda que bloquea cualquier entrenamiento ⬜

**Lo primero, porque es barato y porque sin ello entrenar es peligroso.**

- [ ] **`--vocab_file` en `train_llm`.** Existe `--out_file` para redirigir el
      modelo, pero **no hay forma de redirigir el vocabulario**:
      `tokenizer.Save()` escribe siempre en `release/vocab_cpp.txt`. Entrenar con
      el corpus español lo sobrescribiría con sus 111 símbolos y el modelo de
      Shakespeare —que carga y genera— quedaría inservible, porque el suyo son 53
      caracteres. Y el daño no sería visible: el modelo seguiría cargando y
      decodificaría con la tabla equivocada. Basura silenciosa, no un error.
      **Hoy no se puede entrenar un segundo modelo sin destruir el primero.**
- [ ] **`generate_llm` debe deslizar la ventana, no abortar.** Si la generación
      supera `block_size`, revienta con
      `std::out_of_range: Embedding: token 16 fuera del rango [0, 16)`, porque la
      posición se sale de la tabla de `wpe_`.
- [ ] **Entrenamiento de referencia en español.** Con el corpus ya preparado
      (4.9 M caracteres, ver [tools/corpus/](../tools/corpus/README.md)). No es
      solo un modelo: es la **prueba de humo del canal completo** —nadie ha
      ejecutado un entrenamiento de LLM desde los cambios de estructura— y la
      base contra la que medir el BPE después. Sin ella, el BPE volvería a ser
      una mejora indemostrable, que es exactamente lo que pasó con el autograd.
      Medido: ~14 min con 5000 iteraciones.

## Fase 13 — Cerrar el motor ⬜ (corresponde a 0.5 y 0.6)

**Contra lo que parecía, aquí faltan dos cosas, no dos fases.** El inventario
contra el código: `strides`, `views`, `broadcasting`, reducciones por eje,
`Transpose`, acumulación de gradientes, `Reshape` y `Transpose` derivables,
backward con broadcasting y `Conv2DVar` **ya existen**. Queda esto:

- [ ] **`Concat` y su derivada.** Es la pieza que más desbloquea de todo el
      roadmap por lo poco que cuesta: sin ella no hay skips de U-Net, y son la
      mitad de la arquitectura.
- [ ] **`Backward(salida, gradiente_externo)`.** Hoy `Backward` exige una raíz
      escalar y siembra el gradiente él mismo, así que propagar un `dout`
      concreto obliga al rodeo `Sum(Mul(salida, dout))` —que es literalmente lo
      que hacen `LinearAutograd` y `EmbeddingAutograd`— y materializa un tensor
      del tamaño de la salida entera. Ése es el coste fijo que hace que la
      versión por grafo sea 6× más lenta incluso con tablas pequeñas.
- [ ] **`dtype`.** Lo último de la lista y lo menos urgente: hoy todo es `float`
      y no hay ningún caso de uso que lo exija. Se anota para no perderlo.

**Criterio de salida: cuando esto funcione, parar.** El objetivo del autograd es
permitir construir arquitecturas nuevas, no reimplementar NeuralSuite por
segunda vez.

## Fase 14 — Vocabulario neural compartido ⬜ (corresponde a 0.7)

**El cambio de filosofía: dejar de acumular demos y tener piezas reutilizables.**
`RMSNorm` no pertenece a GPT ni `GroupNorm` a la difusión; son capacidades del
framework que después usan LLM, visión y OCR.

Hay ya: `LayerNorm`, `Conv2D`, `MaxPool2D`, `MultiHeadAttention`, `Residual`,
`ReLU`, `GELU`, `Sigmoid`. Faltan seis, en este orden —las dos primeras porque
desbloquean el transformer moderno, que es el examen principal—:

- [ ] `RMSNorm`
- [ ] `SiLU`
- [ ] `GroupNorm`
- [ ] `Upsample2D`
- [ ] `Downsample2D`
- [ ] `CrossAttention` — `Q` del latente, `K` y `V` del condicionamiento. Es la
      pieza que conecta lenguaje y visión, y la que convierte dos modelos
      separados en un sistema multimodal.

Cada una con su paridad contra PyTorch, como el resto.

## Fase 15 — Transformer moderno ⬜ (corresponde a 0.8)

**Sigue siendo el examen principal del framework, y está a mitad.** Ya hay
KV-Cache (medido: 9.6×, misma secuencia exacta), recorte de gradiente,
planificador de tasa de aprendizaje, checkpoint/resume, paridad de entrenamiento
con PyTorch y ahora corpus en español.

- [ ] **RoPE** — los cinco requisitos están en
      [FUTURE_PLAN_KVCACHE_ROPE.md](FUTURE_PLAN_KVCACHE_ROPE.md). Ojo con el
      error clásico: al usar el KV-Cache hay que rotar con la posición absoluta,
      no con el índice dentro de la caché.
- [ ] **SwiGLU**
- [ ] **GQA**
- [ ] **Perplejidad** como métrica, sobre validación y sobre prueba.

**Criterio de salida:** entrenar un transformer pequeño de verdad y demostrar que
NeuralSuite y PyTorch siguen trayectorias de entrenamiento equivalentes. Eso es
más fuerte que la paridad de un solo paso que ya existe.

## Fase 16 — Datos ⬜

**Fase que el plan original no tenía y sin la cual las siguientes no arrancan.**
Hoy **no existe ningún cargador de datasets**: nada lee MNIST ni CIFAR. El mismo
agujero que costó medio día descubrir en el LLM —donde el corpus eran 3.2 KB de
Shakespeare— está intacto en visión.

- [ ] **Lector de MNIST** con verificación de checksum y formato.
- [ ] **`DataLoader`**: lotes, barajado y partición reproducibles.

## Fase 17 — Difusión de verdad ⬜ (corresponde a 0.9)

La demo actual es un `beta = 0.3f` fijo y un MLP pequeño. Se conserva como
juguete didáctico y se construye la implementación real:

- [ ] `DiffusionSchedule` — `beta[t]`, `alpha[t]`, `alpha_bar[t]`, `q_sample()`,
      `predict_x0()`, `step()`.
- [ ] `SinusoidalTimeEmbedding`.
- [ ] `DDPMSampler` y `DDIMSampler`.
- [ ] `UNet2D` — bloques residuales condicionados por tiempo, skips por `Concat`
      (de ahí la Fase 13), atención en el centro, up/downsampling.
- [ ] **Examen: generar dígitos MNIST reconocibles con DDPM.**

### Por qué MNIST y no CIFAR

Ésta es la corrección más importante al plan propuesto, y sale de medir. Una
pila convolucional al tamaño de la U-Net de un DDPM para CIFAR —**sin** atención,
**sin** skips, **sin** *time embedding*, o sea una cota inferior generosa—:

| | Por paso | 1 época | 200 épocas |
| --- | --- | --- | --- |
| CIFAR-10, canales 64–128 | 150 ms | 7.8 min | **26 horas** |
| MNIST, canales 32–64 | 78 ms | 2.4 min | **8.2 horas** |

Una U-Net real es 5–20× eso, y los DDPM de CIFAR se entrenan 500–800 épocas:
**semanas o meses de CPU**. El plan propuesto colocaba ahí la puerta —«si no
podemos hacer esto, no tiene sentido añadir latent diffusion»— y con CIFAR esa
puerta no se abre nunca. Con MNIST el examen se ejecuta en una tarde.

## Horizonte — sin casillas, deliberadamente

Latent diffusion, compresión perceptual y condicionamiento multimodal son la
**dirección declarada** del proyecto, no un plan con casillas. Se escriben aquí
para que el rumbo esté claro y para no volver a discutirlo desde cero, pero no se
trocean en tareas hasta que la Fase 17 esté cerrada.

La razón es la que ya conoce este documento: **un plan de once fases inalcanzables
es otra lista que diverge**, y este proyecto ya arregló siete.

- **Autoencoder convolucional** (0.10). El actual es `Linear(8,4) → Linear(4,2)`:
  densas, no convolucional, sin latente espacial. Habría que rehacerlo con
  `ConvEncoder`/`ConvDecoder`, latente `[B,C,H/f,W/f]` con `f=4`, y KL.
- **Latent diffusion** (0.12). El experimento interesante sería comparar, con el
  mismo presupuesto, DDPM en píxeles contra DDPM en latente: tiempo de
  entrenamiento, tiempo de muestreo, memoria pico y calidad. Eso ya es un
  experimento y no una demo. Y la aritmética favorece al latente: a `f=4`, un
  32×32 pasa a 8×8, dieciséis veces menos posiciones.
- **Compresión perceptual** (0.13), donde la GAN actual dejaría de ser una demo
  aislada para convertirse en el discriminador por parches del autoencoder.
- **Condicionamiento texto→imagen** (0.14), una vez exista `CrossAttention`.

### Dos dependencias ocultas que hay que resolver antes de prometer nada

- **FID** necesita una Inception preentrenada y **perceptual loss** una VGG.
  Ninguna se puede entrenar en CPU. O se importan pesos de PyTorch —y entonces la
  métrica deja de ser «solo NeuralSuite», que es la premisa del proyecto— o esas
  fases no se pueden evaluar tal como están escritas. Hay que decidirlo antes,
  no al llegar.
- **BF16/FP16** en CPU sin AVX512-BF16 ni AMX no da ganancia, y convertir un
  framework que usa `float` en todas partes es un refactor grande. No es una
  opción de compilador.

---

## Lo que cada capa de verificación encontró

Vale la pena registrarlo, porque justifica el orden del plan: cada capa
detectó defectos que la anterior no podía ver.

| Capa                       | Encontró                                                          |
| -------------------------- | ----------------------------------------------------------------- |
| Lectura del código         | Los dos P0 de gradientes; que `LSTM` no era una LSTM               |
| Gradient checks            | Nada nuevo — pero fijan las correcciones como regresión            |
| Mutación de las pruebas    | Que la prueba de `GraphConv` no comprobaba lo que decía            |
| Prueba de ida y vuelta     | Que el número mágico entraba en el checksum al escribir y no al leer |
| Paridad contra PyTorch     | Detecta errores de semántica que el gradient check no puede ver    |
| Pruebas de dependencia     | De qué entradas depende cada salida: un `BiLSTM` que no mirara hacia atrás sería derivable y consistente consigo mismo |
| Integración continua       | Que Windows nunca compiló; dos rutas que solo existían en una máquina |

El gradient checking compara el código consigo mismo: confirma que el
`Backward` deriva el `Forward` escrito, no que ese `Forward` sea lo que dice
ser. Por eso un gradient check sobre la `LSTM` original **habría pasado**.

Dos lecciones que se repitieron lo bastante como para anotarlas:

**Probar que lo correcto pasa importa tanto como probar que lo incorrecto
falla.** El formato NSF rechazaba bien las seis situaciones inválidas, pero
tampoco aceptaba las válidas: el número mágico se sumaba al checksum al
escribir y no al leer. Solo la prueba de ida y vuelta lo vio.

**El código que funciona en una sola máquina no da síntoma hasta que sale de
ella.** Aparecieron una ruta absoluta del directorio del autor en el arnés de
paridad y rutas `/tmp` en la prueba de serialización, que no existen en
Windows. Ninguna verificación local podía detectarlo.
