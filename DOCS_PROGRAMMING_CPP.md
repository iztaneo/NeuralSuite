# Cómo está escrito NeuralSuite: patrones de C++17

Las decisiones de ingeniería del código, con el porqué de cada una. Casi todas
vienen de un problema real: un defecto que costó encontrar, una medición, o una
diferencia entre plataformas.

Para **cómo encaja todo**, ver [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md);
para **dónde está cada cosa**, [docs/MAPA_DEL_CODIGO.md](docs/MAPA_DEL_CODIGO.md).

---

## 1. Una sola declaración para los pesos

El patrón central del framework, y el que existe por el defecto más caro del
proyecto.

**El problema.** Antes, cada capa exponía **dos listas paralelas** —los pesos y
sus gradientes— que el optimizador recorría por índice. Una capa declaró sus
pesos y olvidó sus gradientes: heredó una lista vacía, y el optimizador acabó
aplicando el gradiente de una capa a los pesos de otra, leyendo fuera de rango al
agotarse la lista más corta. El modelo entrenaba mal sin que nada fallara.

**La solución.** Un `Parameter` reúne el valor y su gradiente en un solo objeto:

```cpp
class Parameter {
 public:
  Tensor& Value();   // el peso
  Tensor& Grad();    // su gradiente
  void ZeroGrad();
};
```

Y cada capa lo **declara una vez** en su constructor:

```cpp
ResBlock2D::ResBlock2D(int c_in, int c_out, int grupos)
    : norm1_(grupos, c_in), conv1_(c_in, c_out, 3, 1, 1), ... {
  Register(&norm1_, "norm1");
  Register(&conv1_, "conv1");
  // ...
}
```

De esa única declaración salen **las tres listas**:

```cpp
[[nodiscard]] std::vector<Tensor*> GetParameters() {
  std::vector<Tensor*> out;
  for (Parameter* p : Parameters()) out.push_back(&p->Value());
  return out;
}
```

`GetParameters()` y `GetGradients()` **no son virtuales**: una capa no puede
sobrescribir una y olvidar la otra, porque las dos se derivan de `Parameters()`.
El error de antes ya no se puede escribir.

`Register()` también acepta submódulos, así que un modelo es un **árbol**, y de
ahí salen los nombres con ruta (`blocks.0.attn.weight`) que usa el formato de
pesos.

---

## 2. RAII con almacenamiento compartido

**Lo que no se hace.** `Tensor` no tiene `float*` con `new[]` y `delete[]`. Esa
versión obligaba a escribir a mano las cuatro operaciones de copia y movimiento,
y cualquier olvido era una fuga o un doble borrado.

**Lo que hace.** El bloque de memoria es un `std::vector<float>` con propiedad
compartida:

```cpp
using Storage = std::vector<float>;
std::shared_ptr<Storage> storage_;
```

Eso da liberación automática **y** algo que un `float*` no daba: **varios
tensores pueden ver la misma memoria con formas distintas**.

```cpp
[[nodiscard]] Tensor View(const std::vector<int>& new_shape) const;
[[nodiscard]] bool SharesStorageWith(const Tensor& other) const;
```

Aplanar `[B, T, C]` a `[B·T, C]` antes de una capa densa pasa de ser una reserva
más un `memcpy` a ser gratis. Medido sobre un paso del GPT: la memoria reservada
bajó de 118 MB a 72.6 MB.

**La regla que hay que tener presente:** **copiar un `Tensor` copia los datos**;
`View()` los comparte. La diferencia importa: `DataLoader::Partir()` devolvía
particiones que **duplicaban** el conjunto entero, porque pasaba un *lvalue* a un
parámetro por valor y el constructor de copia reserva memoria nueva. Nada
fallaba —los datos eran correctos— pero partir MNIST gastaba 180 MB de más.

Por eso `SharesStorageWith()` es público: hace la promesa **comprobable**, y hay
una prueba que la comprueba.

---

## 3. Paralelismo que no cambia el resultado

**Lo que no se usa.** OpenMP. Los `#pragma omp` se ignoraban silenciosamente en
macOS con AppleClang, y todo corría en un solo hilo sin que nadie lo notara. El
paralelismo lo aporta `include/parallel.h` con la biblioteca estándar.

**El reparto es dinámico.** El trabajo se divide en más bloques que hilos y cada
uno toma el siguiente al desocuparse, porque los núcleos no son iguales: en un
Apple M5 conviven núcleos de rendimiento y de eficiencia, y repartir en partes
iguales dejaría a todos esperando al más lento.

```cpp
parallel::ParallelFor(cuenta, minimo_por_hilo, [&](int ini, int fin) {
  for (int i = ini; i < fin; ++i) { /* ... */ }
});
```

**La regla que lo hace verificable:** cada hilo escribe posiciones que **ningún
otro toca**. Sin reducción entre hilos no hay sumas en orden distinto, y el
resultado es **idéntico bit a bit** con 1 hilo que con 10. Eso es lo que permite
comparar contra PyTorch y obtener exactamente los mismos números.

Cuando una operación necesita reducir sobre un eje, se hace en **dos pasadas
sobre ejes distintos** en vez de compartir acumuladores: en `RMSNorm`, `dx` se
calcula por filas y `dgamma` por columnas.

**Anidar está contemplado.** Si ya se está dentro de una región paralela, el
reparto se ejecuta en serie: anidar agotaría los hilos y podría bloquear.

---

## 4. Fallar pronto y decir por qué

Dos mecanismos, según de quién sea la culpa:

**Excepciones para errores de programación.** Una forma que no cuadra, un índice
fuera de rango, un argumento imposible. Hay 109 `throw std::invalid_argument` en
la biblioteca, y cada mensaje dice **qué llegó y qué se esperaba**:

```cpp
throw std::invalid_argument(
    "UNet2D: alto y ancho deben ser multiplos de 4 (dos bajadas de factor 2); "
    "llegaron " + std::to_string(x.Shape()[2]) + "x" + std::to_string(x.Shape()[3]));
```

**Un `Result` para errores esperables**, como que un archivo no exista o esté
corrupto:

```cpp
struct Result { bool ok; std::string error; };
```

Leer un archivo ajeno no es un error de programación, así que no lanza: devuelve
el motivo para que quien llama decida.

**La validación de índices se paga solo en `Debug`.** `Tensor::operator[]`
comprueba el rango bajo `#ifndef NDEBUG`, así que el build de desarrollo atrapa
desbordamientos y el de producción no paga por ello. Por eso el CI compila y
prueba en **los dos modos**.

---

## 5. Portabilidad, aprendida a golpes

Tres diferencias entre plataformas que encontró el CI y que el código ahora
respeta:

- **No existe `/tmp` en Windows.** Nueve pruebas lo daban por hecho. Ahora hay
  una única forma de pedir una ruta temporal, sobre
  `std::filesystem::temp_directory_path()`.
- **`std::rename` falla si el destino existe en Windows**, mientras que en POSIX
  lo reemplaza. Como los checkpoints se escriben en temporales y se mueven a su
  sitio, **solo se guardaba el primero** de cada entrenamiento. Se usa
  `std::filesystem::rename`, que por norma reemplaza.
- **`long` es de 32 bits en Windows.** Desplazar iteradores con
  `static_cast<long>` desbordaría con imágenes grandes: se usa `std::ptrdiff_t`.

Ninguna se manifiesta en macOS ni en Linux. Por eso el CI compila en las tres
plataformas y en los dos modos de build.

---

## 6. El binario refleja el árbol de archivos

`Makefile` y `CMakeLists.txt` recogen los fuentes con `wildcard` y
`GLOB_RECURSE`. Un archivo nuevo entra solo.

No es pereza: **la lista escrita a mano se quedó corta cuatro veces**, y una de
ellas dejó fuera un programa entero durante días. Que el build dependa del árbol
de archivos y no de una lista es una fuente menos de divergencia.

Por la misma razón, `include/` contiene la interfaz y `src/` los cuerpos, con las
mismas subcarpetas. Tres archivos son la excepción y lo explican dentro:
`serialization.h` y `autograd.h` por las plantillas, y `parallel.h` porque su
protocolo entre hilos no se entiende partido en dos.

---

## 7. La verificación como parte del diseño

Dos patrones que existen solo para poder comprobar el código:

**Implementaciones de referencia.** `Conv2DReference`, `LSTMReference` y
`MultiHeadAttentionReference` son las versiones literales, lentas y obvias. No
las usa nadie para entrenar: existen para que una prueba compruebe que la versión
rápida calcula exactamente lo mismo.

**Fuentes de azar inyectables.** Los muestreadores de difusión y el latente
gaussiano **reciben el ruido**, no lo generan:

```cpp
using FuenteRuido = std::function<void(int paso, Tensor* destino)>;
```

Si cada uno generase el suyo, dos implementaciones correctas darían trayectorias
distintas y la comparación contra PyTorch no podría distinguirlas de dos
incorrectas.

Por el mismo motivo, los muestreadores toman el modelo como una función
`(x_t, t) -> eps` en vez de una `UNet2D`: así se pueden probar con un **oráculo
analítico** cuya respuesta exacta se conoce, sin modelo entrenado de por medio.

---

## 8. Convenios de estilo

- **Nombres en español** para lo propio del dominio (`Codificador`,
  `GaussianaDiagonal`, `RutaArchivada`) y **en inglés** para lo que imita una API
  conocida (`Forward`, `Backward`, `GetParameters`). Los comentarios, en español.
- **Los comentarios explican el porqué, no el qué.** Si un comentario repite lo
  que dice el código, sobra; si explica por qué se eligió así, o qué error evita,
  se queda.
- **`[[nodiscard]]` en lo que devuelve un valor que no debe ignorarse.** Ya
  atrapó un descarte silencioso en una prueba.
- **Cada capa guarda lo que su `Backward` necesitará.** Miembros como `h1_pre_`
  no son estado accidental: son la mitad del cálculo.
- **`Backward` acumula el gradiente de los pesos, nunca lo sobrescribe.** Es lo
  que permite que un peso reciba gradiente por dos caminos, como en un bloque
  residual o en el embedding compartido del GPT.
