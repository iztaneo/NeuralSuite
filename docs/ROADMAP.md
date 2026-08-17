# Hoja de ruta: Core Correctness & Architecture

Estado del plan que llevó a NeuralSuite de una colección de implementaciones a
un framework con sus operaciones verificadas. Cada punto cerrado enlaza el
commit que lo cerró.

El principio que ordena el plan: **cada operación debe ser correcta,
comprobable y reutilizable antes de añadir la siguiente arquitectura.**

---

## Estado actual

| Área                              | Estado                                                        |
| --------------------------------- | ------------------------------------------------------------- |
| Corrección de gradientes          | ✅ verificada por diferencias finitas y contra PyTorch          |
| Robustez de `Tensor`              | ✅ formas validadas, sin estados inválidos, vistas sin copia    |
| Testing                           | ✅ 17 pruebas, validadas por mutación, con código de salida     |
| Portabilidad                      | ✅ Linux (GCC/Clang), macOS y Windows en CI, Debug y Release    |
| Serialización                     | ❌ sin cabecera ni versión: un checkpoint incompatible da basura |
| API para terceros                 | ⚠️ `Parameter`/`Module` pendientes                              |
| Autograd                          | ❌ cada capa implementa su backward a mano                      |
| Rendimiento                       | ⚠️ correcto pero sin optimizar; el cómputo domina               |

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

## Fase 03 — `Parameter` y `Module` ⬜

- [ ] `Parameter { Tensor data; Tensor grad; }` en lugar de dos listas
      paralelas. Haría **imposible por diseño** el defecto de la Fase 01, que
      hoy solo está atrapado por una guarda en el optimizador.
- [ ] Base `Module` con registro automático de submódulos, para eliminar las
      cascadas manuales de `GetParameters()`.

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

## Fase 05 — Autograd ⬜

- [ ] Primitivas con derivada automática (add, mul, matmul, sum, mean, exp, log,
      tanh, reshape, transpose), y después softmax, LayerNorm, gather y
      convolución.

Reduce la superficie de error: hoy cada capa implementa su backward a mano, que
es exactamente donde aparecieron los dos defectos P0.

## Fase 06 — Serialización ⬜ ← siguiente

- [ ] Formato con número mágico, versión, configuración de la arquitectura y
      tensores con nombre.

Hoy `SaveWeights`/`LoadWeights` vuelcan y leen floats crudos sin cabecera: un
checkpoint que no corresponde a la arquitectura configurada **se carga sin dar
error**, con datos truncados o basura. Es la última vía de fallo silencioso
abierta. El contenedor `NSPARITY` de [`tools/parity/`](../tools/parity/README.md)
ya funciona y sirve de base.

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
- [ ] RNG controlable (`manual_seed`) y estado por hilo.

## Fase 08 — Tokenizador ⬜

- [ ] Token `<UNK>` explícito: hoy un carácter desconocido se convierte en el
      token 0, que puede ser un carácter válido del vocabulario.
- [ ] `ByteTokenizer` y BPE propio. El actual trabaja sobre `char`, que en C++
      es un byte: el texto UTF-8 queda fragmentado.

## Fase 09 — Rendimiento ⬜

- [ ] GEMM con blocking, tiling y SIMD; kernel optimizado de Conv2D conservando
      el actual como oráculo de referencia.
- [ ] KV cache contiguo (hoy `vector<vector<float>>`).
- [ ] Decidir sobre RoPE: `ApplyRoPE()` existe pero no se invoca en ningún
      forward, lo que sugiere una capacidad que el modelo no tiene.

## Fase 10 — Ecosistema ✅ (parcial)

`cb0c07d`, `49ef343`

- [x] CI en Linux (GCC y Clang), macOS y Windows, en Debug y Release; suite
      numérica, demos, entrenamiento de extremo a extremo, paridad con PyTorch y
      sanitizers.
- [x] Comparación contra la implementación de referencia en PyTorch.
- [ ] Benchmarks separados de las pruebas.
- [ ] `softmax(x, dim)` y broadcasting.
- [ ] Grupos de parámetros explícitos en AdamW, en vez de inferir el decay del
      rango del tensor.

## Pendiente aparte — OCR ⬜

- [ ] `CRNNModel` no decodifica imágenes: no hay lector de PNG/JPG y la
      arquitectura (Conv + Linear) no corresponde a la de referencia en PyTorch,
      que usa un BiLSTM. Hasta entonces la demo declara explícitamente que corre
      sobre datos sintéticos.

---

## Lo que cada capa de verificación encontró

Vale la pena registrarlo, porque justifica el orden del plan: cada capa
detectó defectos que la anterior no podía ver.

| Capa                       | Encontró                                                        |
| -------------------------- | --------------------------------------------------------------- |
| Lectura del código         | Los dos P0 de gradientes; que `LSTM` no era una LSTM             |
| Gradient checks            | Nada nuevo — pero fijan las correcciones como regresión          |
| Mutación de las pruebas    | Que la prueba de `GraphConv` no comprobaba lo que decía          |
| Paridad contra PyTorch     | Detecta errores de semántica que el gradient check no puede ver  |
| Integración continua       | Que Windows nunca compiló; una ruta absoluta en un script        |

El gradient checking compara el código consigo mismo: confirma que el
`Backward` deriva el `Forward` escrito, no que ese `Forward` sea lo que dice
ser. Por eso un gradient check sobre la `LSTM` original **habría pasado**.
