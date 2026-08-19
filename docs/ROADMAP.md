# Hoja de ruta: Core Correctness & Architecture

Estado del plan que llevó a NeuralSuite de una colección de implementaciones a
un framework con sus operaciones verificadas. Cada punto cerrado enlaza el
commit que lo cerró.

El principio que ordena el plan: **cada operación debe ser correcta,
comprobable y reutilizable antes de añadir la siguiente arquitectura.**

---

## Estado actual

**52 puntos cerrados, 8 pendientes.** Lo que queda con algo incorrecto es el OCR; el resto son mejoras sobre código ya verificado.

| Fase                       | Estado           | Fase              | Estado       |
| -------------------------- | ---------------- | ----------------- | ------------ |
| 00 Confianza y limpieza    | ✅               | 06 Serialización  | ✅           |
| 01 Corrección crítica      | ✅               | 07 Runtime y build| ✅           |
| 02 Tensor Core             | ✅ parcial       | 08 Tokenizador    | ✅ parcial   |
| 03 `Parameter` y `Module`  | ✅               | 09 Rendimiento    | ✅ parcial   |
| 04 Verificación matemática | ✅               | 10 Ecosistema     | ✅ parcial   |
| 05 Autograd                | ✅ parcial       | OCR (aparte)      | ⬜           |

| Área                              | Estado                                                        |
| --------------------------------- | ------------------------------------------------------------- |
| Corrección de gradientes          | ✅ verificada por diferencias finitas y contra PyTorch          |
| Robustez de `Tensor`              | ✅ formas validadas, sin estados inválidos, vistas sin copia    |
| Testing                           | ✅ 24 pruebas, validadas por mutación, con código de salida     |
| Portabilidad                      | ✅ Linux (GCC/Clang), macOS y Windows en CI, Debug y Release    |
| Serialización                     | ✅ formato NSF con versión, metadatos y checksum                 |
| API para terceros                 | ✅ `Parameter` y `Module` con registro automático                |
| Autograd                          | ✅ motor, primitivas y LayerNorm compuesta; capas sin migrar     |
| Rendimiento                       | ⚠️ paralelizado (2.2x); falta blocking y SIMD en el GEMM        |

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

## Fase 02 — Tensor Core ✅ (parcial)

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
- [ ] Strides, `Transpose` como vista y `Contiguous()`. Aplazado: hoy ninguna
      operación consume vistas no contiguas, así que sería refactor sin
      beneficio. Gana sentido junto con el autograd.

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
- [ ] Migrar las capas existentes a las primitivas. Deliberadamente aparte: el
      motor debía estar verificado antes de reescribir sobre él nada que ya
      funciona y está comprobado contra PyTorch.
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
- [ ] `gather` y convolución como primitivas.

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

## Fase 07 — Runtime y build portable ✅

`cb0c07d`, `49ef343`

- [x] **El build de CMake no funcionaba en macOS.** `find_package(OpenMP
      REQUIRED)` falla con AppleClang, así que la instrucción principal del
      README nunca se había ejecutado en una de las tres plataformas que
      anuncia. OpenMP pasa a ser opcional.
- [x] `-march=native` y `-ffast-math` dejan de aplicarse siempre. La primera
      producía binarios que podían no arrancar en otra CPU; la segunda relaja
      IEEE-754 justo donde se verifican gradientes.
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

## Fase 09 — Rendimiento ⬜

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
      de rendimiento: `MatMul` pasa de 35 a 192 GFLOP/s, y el paso de
      entrenamiento de 119 ms a unos 29 ms.
- [x] **Reparto dinámico en lugar de trozos iguales.** Con un trozo por hilo, el
      tiempo lo marca el más lento, y los núcleos rara vez son iguales: un
      Apple M5 mezcla núcleos de rendimiento y de eficiencia, y en un servidor
      dos hilos pueden compartir el mismo núcleo físico. Repartir bloques que
      cada hilo toma al desocuparse sube la aceleración de 3.7x a 5.3x y el paso
      de entrenamiento de 37 ms a 29 ms. Es justo el margen que separaba al
      reparto propio de OpenMP. El resultado es idéntico bit a bit,
      porque cada hilo escribe filas disjuntas y no hay reducción que altere el
      orden de las sumas.
- [ ] GEMM con blocking, tiling y SIMD; kernel optimizado de Conv2D conservando
      el actual como oráculo de referencia.
- [ ] KV cache contiguo (hoy `vector<vector<float>>`).
- [x] **RoPE decidido**: se retira `ApplyRoPE()`, que ningún forward llamaba y
      sugería una capacidad que el modelo no tiene. La posición sigue llegando
      por embeddings aprendidos. RoPE continúa planteado en
      `FUTURE_PLAN_KVCACHE_ROPE.md`; implementarlo exige rotar Q y K, propagar
      por esa rotación con su gradient check, y actualizar la implementación de
      referencia en PyTorch para que la comparación siga siendo válida.

## Fase 10 — Ecosistema ✅ (parcial)

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

- [ ] `CRNNModel` no decodifica imágenes: no hay lector de PNG/JPG y la
      arquitectura (Conv + Linear) no corresponde a la de referencia en PyTorch,
      que usa un BiLSTM. Hasta entonces la demo declara explícitamente que corre
      sobre datos sintéticos.

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
