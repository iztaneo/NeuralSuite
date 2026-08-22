# Hoja de ruta: Core Correctness & Architecture

Estado del plan que llevó a NeuralSuite de una colección de implementaciones a
un framework con sus operaciones verificadas. Cada punto cerrado enlaza el
commit que lo cerró.

El principio que ordena el plan: **cada operación debe ser correcta,
comprobable y reutilizable antes de añadir la siguiente arquitectura.**

---

## Lo que falta

Ocho frentes, y solo uno de ellos es algo que hoy esté **mal** —el OCR, ya en
curso—; el resto son mejoras sobre código ya verificado. Esa distinción es la
que ordena el trabajo, más que el número de fase.

### Incorrecto hoy

| | Fase | Qué pasa |
| --- | --- | --- |
| **OCR** | aparte | **Lee texto impreso real**: 3.4% de error de carácter sobre una página de libro que nunca vio, frente al 53% de hace dos días y al 0.1% de Tesseract. Lo que queda del error son sobre todo caracteres fuera del vocabulario —acentos y puntuación—. Ver [Pendiente aparte — OCR](#pendiente-aparte--ocr-). |

### Mejoras sobre lo verificado

| | Fase | Nota |
| --- | --- | --- |
| `gather` y convolución como primitivas de autograd | 05 | Continúa el motor; trabajo de tamaño propio. |
| Tesseract como referencia en `tools/` | 10 | Mismo papel que PyTorch y Pillow: instrumento de medida, nunca dentro de la biblioteca. Sin él, «54% de error» no dice si estamos lejos o cerca. Ya está instalado en la máquina de desarrollo. |
| Migrar las capas al autograd | 05 | Sustituiría código verificado contra PyTorch por código sin comprobar. El cambio más grande y el más arriesgado. |
| BPE propio | 08 | Reduciría la longitud de las secuencias; hoy un carácter no ASCII ocupa varios tokens. |
| Estado del generador por hilo | 07 | El RNG es global; no es seguro usarlo desde varios hilos. Hoy nadie lo hace. |

### Rendimiento

Las tres capas con bucles profundos que no llamaban a `MatMul` —`Conv2D`, la
celda recurrente y la atención— ya están reformuladas, con **55×**, **16×** y
**17.6×** respectivamente. El barrido está cerrado; lo que queda es de otra
naturaleza.

| | Fase | Nota |
| --- | --- | --- |
| Intrínsecos SIMD por arquitectura | 09 | Hoy el bucle interno lo autovectoriza el compilador. |
| KV cache contiguo | 09 | Solo afecta a la generación token a token. |

---

## Estado por área

| Área                              | Estado                                                        |
| --------------------------------- | ------------------------------------------------------------- |
| Corrección de gradientes          | ✅ verificada por diferencias finitas y contra PyTorch          |
| Robustez de `Tensor`              | ✅ formas validadas, sin estados inválidos, vistas sin copia    |
| Testing                           | ✅ 24 pruebas, validadas por mutación, con código de salida     |
| Portabilidad                      | ✅ Linux (GCC/Clang), macOS y Windows en CI, Debug y Release    |
| Serialización                     | ✅ formato NSF con versión, metadatos y checksum                 |
| API para terceros                 | ✅ `Parameter` y `Module` con registro automático                |
| Autograd                          | ✅ motor, primitivas y LayerNorm compuesta; capas sin migrar     |
| Rendimiento                       | ✅ 4.5x medido; falta SIMD explícito y kernel de Conv2D          |

| Fase                       | Estado           | Fase              | Estado       |
| -------------------------- | ---------------- | ----------------- | ------------ |
| 00 Confianza y limpieza    | ✅               | 06 Serialización  | ✅           |
| 01 Corrección crítica      | ✅               | 07 Runtime y build| ✅ parcial   |
| 02 Tensor Core             | ✅               | 08 Tokenizador    | ✅ parcial   |
| 03 `Parameter` y `Module`  | ✅               | 09 Rendimiento    | ✅ parcial   |
| 04 Verificación matemática | ✅               | 10 Ecosistema     | ✅           |
| 05 Autograd                | ✅ parcial       | OCR (aparte)      | 🔄 en curso  |

El recuento sale del propio documento, no de un número escrito a mano —que ya
divergió tres veces—:

```bash
echo "cerrados: $(grep -c '^- \[x\]' docs/ROADMAP.md)  pendientes: $(grep -c '^- \[ \]' docs/ROADMAP.md)"
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
- [ ] KV cache contiguo (hoy `vector<vector<float>>`).
- [x] **RoPE decidido**: se retira `ApplyRoPE()`, que ningún forward llamaba y
      sugería una capacidad que el modelo no tiene. La posición sigue llegando
      por embeddings aprendidos. RoPE continúa planteado en
      `FUTURE_PLAN_KVCACHE_ROPE.md`; implementarlo exige rotar Q y K, propagar
      por esa rotación con su gradient check, y actualizar la implementación de
      referencia en PyTorch para que la comparación siga siendo válida.

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

- [x] **JPEG**, secuencial de línea base y progresivo. Huffman, cuantización,
      transformada inversa e interpolación de crominancia. Es el único formato
      cuya salida **no está especificada bit a bit**: la norma fija requisitos
      de precisión para la transformada (T.83), no un resultado exacto, así que
      la comparación con libjpeg necesita un criterio estadístico y la
      transformada se verifica aparte contra su definición matemática. El modo
      aritmético, el sin pérdida y el de 12 bits se rechazan diciendo cuál es.

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
