# Capas derivadas por el grafo

> Estado: **funcionalidad adicional, no conectada a nada.** Ningún modelo la usa.
> `gpt.cpp`, el CRNN y el entrenamiento de OCR siguen usando las capas de
> siempre. Lo único que llama a este código son las pruebas.

Este documento explica qué hacen las cuatro piezas nuevas, cuánto cuestan
—medido, no estimado—, qué encontraron —nada— y por dónde seguir si alguien
las retoma.

---

## 1. Qué hay

### `Gather(tabla, indices)` — `include/autograd.h`

Toma filas de una tabla `[filas, dimensión]` por índice. La salida añade la
dimensión al final: con `indices` de forma `[B, T]` devuelve `[B, T, dim]`.
Valida que cada índice caiga dentro de la tabla.

Hacia atrás, **cada fila recibe la suma de los gradientes de todas las
posiciones que la usaron**. Esa suma es la razón de que no se pueda escribir
como una copia: en una secuencia real el mismo token aparece muchas veces, y
quedarse con el último gradiente en vez de sumarlos produce un resultado que
parece razonable y está mal. Es la operación que `Embedding` hacía a mano.

### `Conv2DVar(entrada, pesos, stride, padding)` — `include/autograd.h`

Convolución 2D derivable. `entrada` es `[lote, canales, alto, ancho]` y `pesos`
`[canales_salida, canales_entrada, k, k]`.

**No lleva sesgo**: hay que sumarlo aparte con `Add`, que el motor ya sabe
derivar. Por dentro reutiliza `Conv2D` —la versión rápida, comprobada contra
`Conv2DReference` y contra PyTorch— en vez de recomponer la convolución con
primitivas más pequeñas. Envolver lo que ya funciona evita perder el
rendimiento del `im2col` y mantiene el gradiente saliendo de código verificado.

### `LinearAutograd` — `include/layers/linear_autograd.h`

`Y = X · W + b`. Intercambiable con `Linear`: mismo constructor, misma
inicialización Xavier, mismos parámetros en el mismo orden. Toda la capa son
dos líneas:

```cpp
salida_ = autograd::Add(autograd::MatMulVar(entrada_, peso_), sesgo_);
```

El backward no lo escribió nadie.

### `EmbeddingAutograd` — `include/layers/embedding_autograd.h`

Intercambiable con `Embedding`. Toda la capa es una llamada a `Gather`.

Resuelve correctamente el caso que usa el GPT sin código que lo trate: el
embedding de posición se calcula con forma `[1, T]` y su gradiente llega como
`[B, T, D]`, así que cada posición debe sumar las `B` contribuciones. A
`EmbeddingAutograd` le sale del broadcasting de `Mul` y de la reducción que
hace el motor al propagar. `Embedding` lo consigue con un `% num_cached` en su
bucle que a primera vista parece defensivo y en realidad **es carga
estructural**: quitarlo pone la prueba en rojo.

---

## 2. Por qué existen en pareja en vez de sustituir a las originales

Los dos defectos P0 del proyecto fueron gradientes escritos a mano. Un backward
que nadie escribió no puede tener esa clase de error, así que la versión del
grafo sirve de **oráculo independiente**: llega al mismo número por un camino
que no comparte código ni fórmula con el original, y sin necesitar PyTorch
instalado.

El proyecto ya mantiene tres parejas así —`Conv2DReference`, `LSTMReference`,
`MultiHeadAttentionReference`— y ninguna ha divergido nunca, mientras que las
seis listas hechas a mano que nadie comparaba divergieron todas.

**La condición que hace segura la duplicación no es la disciplina: es que exista
una prueba que enfrente a la pareja.** Si alguien añade una quinta pareja sin
esa prueba, está creando la séptima lista.

---

## 3. Lo que cuestan — medido

Reconstruido limpio, `-O2`, promedio de 20 pasadas forward+backward.

| | A mano | Por grafo | |
|---|---|---|---|
| `Linear` 768×768, 2048 filas | 45 ms | 120 ms | **2.7× más lenta** |
| `Embedding` vocab 50257, dim 768 | 11.8 ms | 241 ms | **20× más lenta** |

El coste del embedding **escala con el tamaño de la tabla, no con cuántos
tokens se miran** (constante, 2048 en las tres filas):

| vocabulario | tabla | a mano | por grafo | factor |
|---|---|---|---|---|
| 512 | 2 MB | 10.4 ms | 62.9 ms | 6.1× |
| 4096 | 13 MB | 10.3 ms | 75.7 ms | 7.4× |
| 50257 | 154 MB | 11.4 ms | 230.1 ms | 20.2× |

Son dos costes distintos superpuestos:

- **Uno fijo**, proporcional al tamaño de la *salida*: cerrar el grafo con
  `Sum(Mul(salida_, dout))` materializa un tensor del tamaño de la salida
  entera. Es lo que explica el 6× incluso con una tabla de 2 MB.
- **Uno que escala con la *tabla***: cada `Forward` crea la hoja del grafo
  copiando la tabla completa y reservando un gradiente del mismo tamaño; el
  backward reserva otro tensor de tabla completa, lo recorre entero para
  ponerlo a cero, lo acumula, y al final se copia otra vez al `Parameter`. Son
  cuatro o cinco recorridos de 154 MB por paso para tocar 2048 filas.

---

## 4. Lo que encontraron: nada

Las tres pruebas pasaron a la primera. Ninguna capa existente estaba mal.

Vale la pena registrarlo con precisión, porque durante el trabajo afirmé que la
prueba nueva cubría un caso descubierto —el `%` de `Embedding::Backward`— y
**era falso**: al romper esa línea la suite unitaria pasaba igual, pero la
paridad contra PyTorch sí lo cazaba (`wpe.weight`, error `1.000e+00`). Lo único
que aporta la prueba nueva es mover esa cobertura de la paridad —que necesita
PyTorch y se corre a mano— a la suite unitaria, que corre siempre.

**Saldo honesto a día de hoy:** ~330 líneas, dos capas más lentas que nadie va a
usar para entrenar, cero defectos encontrados, y una cobertura que ya existía en
otro sitio. El valor es prospectivo: un oráculo permanente para los cambios que
vengan.

---

## 5. Límites conocidos

- `Gather` exige que la tabla sea exactamente 2D.
- `Conv2DVar` no lleva sesgo; hay que sumarlo aparte.
- `Conv2DVar` **construye un `Conv2D` nuevo en cada llamada**, con la reserva de
  memoria que eso implica. Para una prueba da igual; para entrenar, no.
- Las dos capas rehacen las hojas del grafo en cada `Forward`, porque los
  parámetros cambian tras cada paso del optimizador. Es correcto y es caro.
- Ninguna de las dos está paralelizada.
- El grafo retiene los tensores intermedios de cada pasada hasta que se suelta
  la raíz. Con lotes grandes eso es memoria que la versión a mano no gasta.

---

## 6. Si alguien lo retoma

Por orden de lo que más devuelve por lo que cuesta:

1. **Evitar copiar la tabla entera.** Es el 20×. Un gradiente disperso —lista de
   `(fila, vector)` en vez de una tabla completa casi toda a cero— convertiría
   el backward del embedding en proporcional a los tokens vistos, que es lo que
   ya hace la versión a mano. Toca `Gather` y `Variable::AccumulateGrad`.
2. **Sembrar el gradiente sin materializarlo.** Hoy `Backward` exige una raíz
   escalar, y las capas cierran el grafo con `Sum(Mul(salida, dout))` para
   propagar un `dout` concreto. Un `Backward(raiz, semilla)` que aceptara el
   gradiente inicial quitaría ese tensor intermedio y el coste fijo con él.
3. **Reutilizar el `Conv2D` interno de `Conv2DVar`** entre llamadas, en vez de
   construirlo cada vez.
4. **Las capas que faltan.** `Conv2D`, `LSTM` y `MultiHeadAttention`. Aviso: las
   tres **ya tienen** su pareja de referencia (`*Reference`), así que el oráculo
   adicional aporta bastante menos que en `Linear` y `Embedding`, que no la
   tenían. Cada pareja cuesta unas 200 líneas.

Y una advertencia que vale más que las cuatro: **el patrón, medido, encuentra
cero defectos y cuesta ~200 líneas por capa.** Antes de seguir migrando conviene
decidir si eso compensa, en vez de completar la fase por completarla.

---

## 7. Cómo comprobar que siguen bien

```bash
make clean && make -j8 && ./bin/test_suite
```

Tests 35, 36 y 37. Las tres comparan contra la capa equivalente y, donde se
puede, contra un valor calculado aparte —para que no baste con que las dos
implementaciones coincidan si ambas se equivocan igual.

Cada una está validada por mutación: romper a propósito el acumulado de
`Gather`, el relleno de `Conv2DVar`, el orden de la suma del sesgo en
`LinearAutograd` o el `%` de `Embedding` pone la prueba correspondiente en rojo.
