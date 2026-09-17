# Teoría e implementación

Para cada concepto: **la idea**, **la matemática**, **dónde está en el código**,
**cómo se comprueba que es correcto** y **de dónde sale**. Pensado para estudiar
con el código delante.

Convenios de notación: $x$ es la entrada, $y$ la salida, $L$ la pérdida,
$\theta$ un peso cualquiera, $\odot$ el producto elemento a elemento. Los
gradientes se escriben $\partial L/\partial x$.

**Índice.** [Fundamentos](#fundamentos) · [Capas](#capas) ·
[Transformer](#transformer) · [Difusión](#difusión) ·
[Difusión latente](#difusión-latente) · [Optimización](#optimización)

---

# Fundamentos

## Retropropagación

**La idea.** Entrenar es ajustar cada peso en la dirección que reduce el error.
Para saber esa dirección hace falta $\partial L/\partial \theta$ de cada peso, y
calcularlo uno a uno sería inviable. La regla de la cadena permite obtenerlos
todos en una sola pasada hacia atrás, reutilizando lo ya calculado.

**La matemática.** Si $y = f(x)$ y conocemos $\partial L/\partial y$, entonces

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

Cada capa solo necesita saber derivar **lo suyo**.

**En el código.** Es el contrato de `include/layer.h`:

```cpp
Tensor Forward(const Tensor& x);      // calcula y, guarda lo necesario
Tensor Backward(const Tensor& dout);  // recibe dL/dy, devuelve dL/dx
```

`Backward` **acumula** el gradiente de los pesos en vez de sobrescribirlo. Eso
permite que un peso reciba gradiente por dos caminos, como en un residuo o en el
embedding compartido del GPT.

**Verificación.** Diferencias finitas en `tests/test_suite.cpp`:
$\partial L/\partial x \approx (L(x+h) - L(x-h)) / 2h$. Si el `Backward` no
deriva ese `Forward`, la prueba se pone roja.

**Referencia.** Rumelhart, Hinton y Williams (1986), Nature 323.

---

## Capa densa

**La idea.** Combinar todas las entradas con pesos aprendidos.

$$y = xW + b$$

**Gradientes.** Con $\partial L/\partial y$ conocido:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^\top, \qquad
\frac{\partial L}{\partial W} = x^\top \frac{\partial L}{\partial y}, \qquad
\frac{\partial L}{\partial b} = \sum_{\text{filas}} \frac{\partial L}{\partial y}$$

Que el gradiente del sesgo sea una **suma** sobre el lote es el error clásico:
olvidarlo hace que el sesgo casi no aprenda.

**En el código.** `include/layers/linear.h`, `src/layers/linear.cpp`. Inicializa
con Xavier: la escala depende de cuántas entradas y salidas tiene la capa, para
que la señal no se apague ni explote al apilar capas.

**Referencia.** Glorot y Bengio (2010), AISTATS.

---

## Softmax y entropía cruzada

**La idea.** Convertir puntuaciones en probabilidades y medir cuánto se equivoca
el modelo al asignárselas.

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad L = -\log p_{\text{correcta}}$$

**El detalle que importa.** La derivada de la combinación es sorprendentemente
simple:

$$\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = \text{correcta}]$$

Por eso ambas se implementan **juntas**: por separado habría que multiplicar dos
jacobianos y se perdería estabilidad numérica.

**En el código.** `include/losses.h`. La pérdida divide entre el número de
ejemplos, así que **el tamaño del lote no cambia la escala del gradiente**.

**Verificación.** Test de gradiente por diferencias finitas, y paridad contra
`nn.CrossEntropyLoss`.

**La otra pérdida: error cuadrático.** Para predecir números en vez de clases:

$$L = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2, \qquad \frac{\partial L}{\partial \hat{y}_i} = \frac{2(\hat{y}_i - y_i)}{N}$$

Es la que usan la difusión —para predecir el ruido— y el autoencoder —para
comparar la reconstrucción con el original—. En esos dos casos el error se **suma**
sobre los píxeles de cada imagen y se promedia en el lote, para que el término
de reconstrucción y la penalización KL queden en la misma escala.

---

# Capas

## Convolución

**La idea.** El mismo detector aplicado en todas las posiciones de la imagen. Un
borde es un borde esté donde esté, así que no tiene sentido aprender un peso
distinto por píxel.

$$y[c_o, i, j] = b[c_o] + \sum_{c_i}\sum_{u}\sum_{v} W[c_o, c_i, u, v]\; x[c_i,\, si+u-p,\, sj+v-p]$$

con $s$ el paso (*stride*) y $p$ el relleno (*padding*).

**La implementación rápida.** Escrito como bucles anidados es lentísimo. El truco
clásico es `im2col`: reorganizar los parches en una matriz y convertir la
convolución en **una multiplicación de matrices**, que es lo que está optimizado.

**En el código.** `src/layers/conv2d.cpp`, función `Im2Col`. Al lado vive
`Conv2DReference`, la versión literal: existe para comprobar que la rápida
calcula exactamente lo mismo.

**Verificación.** Test 30 compara las dos implementaciones en 8 configuraciones,
y comprueba que el resultado es idéntico con uno y con varios hilos.

**Referencias.** LeCun et al. (1998); Chellapilla et al. (2006) para `im2col`.

### Submuestreo por máximo

Reduce la resolución quedándose con el valor más alto de cada ventana:

$$y_{i,j} = \max_{m,n \in [0, k)} x_{\,si+m,\; sj+n}$$

Hacia atrás, **el gradiente va entero a la posición que ganó** y cero al resto:
las demás no influyeron en la salida. Guardar cuál ganó es lo que el `Forward`
tiene que recordar.

En el código: `layers/maxpool2d.h`. Lo usa el OCR; la difusión y el autoencoder
reducen con `Downsample2D`, que promedia en vez de quedarse con el máximo, porque
al generar interesa conservar la intensidad media y no solo el pico.

---

## Normalización

**La idea.** Mantener los valores intermedios en un rango estable. Sin ello, las
activaciones crecen o se apagan al apilar capas y el entrenamiento se vuelve
frágil.

**LayerNorm** normaliza cada ejemplo por separado:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta$$

**RMSNorm** quita la resta de la media, que resulta no ser necesaria:

$$y = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}}$$

**GroupNorm** normaliza por grupos de canales, no por ejemplo entero. Es la que
usan las redes de imagen, porque funciona igual de bien con lotes pequeños.

**El detalle del backward.** El gradiente de una normalización **no** es
simplemente $\gamma$: como $\mu$ y $\sigma$ dependen de todas las entradas, cada
una influye en todas las salidas. Aparecen dos términos de corrección.

**En el código.** `layers/layernorm.h`, `layers/rmsnorm.h`, `layers/groupnorm.h`.
En `RMSNorm`, el backward hace **dos pasadas sobre ejes distintos**: `dx` por
filas y `dgamma` por columnas, para que cada hilo escriba zonas disjuntas y el
resultado no dependa del reparto.

**Referencias.** Ba et al. (2016); Zhang y Sennrich (2019); Wu y He (2018).

---

## Activaciones

**Por qué hacen falta.** Sin una función no lineal entre capas, apilar capas
densas equivale a una sola: la composición de funciones lineales es lineal.

| | Fórmula | Nota |
| --- | --- | --- |
| ReLU | $\max(0, x)$ | Barata; su derivada es 0 o 1 |
| GELU | $x \cdot \Phi(x)$ | Suave; la del Transformer |
| SiLU | $x \cdot \sigma(x)$ | La de la difusión |

**El error clásico.** Derivar SiLU usando su **salida** en vez de su **entrada**.
La fórmula correcta necesita la preactivación:

$$\frac{d}{dx}\big(x\,\sigma(x)\big) = \sigma(x)\,\big(1 + x\,(1 - \sigma(x))\big)$$

Por eso las capas guardan `*_pre_` además de `*_act_`. Se comprobó con una
mutación: derivar con la salida pone la prueba en rojo.

**En el código.** `include/activations.h`, `SiluForward` y `SiluBackward` en
`tensor.h`.

---

## Bloque residual

**La idea.** En vez de que la capa calcule la salida entera, que calcule **lo que
hay que añadir**:

$$y = x + f(x)$$

Hacia atrás, el gradiente se reparte por los dos caminos y **se suman**:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} + \frac{\partial L}{\partial y}\frac{\partial f}{\partial x}$$

El primer término llega **intacto** a las capas de abajo. Eso es lo que permite
entrenar redes profundas.

**En el código.** `layers/residual.h`, `layers/resblock2d.h`,
`diffusion/resblock.h`. El atajo lleva una convolución 1×1 solo si cambian los
canales; si no, es la identidad.

**Verificación.** Anulando la última convolución, la salida debe ser
**exactamente** el atajo. Quitar el gradiente del atajo pone la prueba en rojo
(error 1.86).

**Referencia.** He et al. (2015), arXiv:1512.03385.

---

## LSTM

**La idea.** Procesar una secuencia arrastrando un estado, con **puertas** que
deciden qué recordar y qué olvidar. Resuelve que el gradiente se desvanezca al
propagarlo por muchos pasos.

$$i = \sigma(\cdot),\quad f = \sigma(\cdot),\quad g = \tanh(\cdot),\quad o = \sigma(\cdot)$$
$$c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)$$

La puerta de olvido $f$ multiplica el estado anterior: si vale 1, el estado pasa
intacto, que es el camino por el que el gradiente sobrevive.

**En el código.** `include/layers/lstm.h` (259 líneas de interfaz),
`src/layers/lstm.cpp` (430 de implementación). Las cuatro puertas se calculan de
una vez con una sola multiplicación de matrices y luego se parten.

**Verificación.** Diferencias finitas sobre los cuatro tensores de parámetros y
sobre la entrada, más paridad contra `nn.LSTM` de PyTorch.

**Referencia.** Hochreiter y Schmidhuber (1997).

---

# Transformer

## Atención

**La idea.** Cada posición mira a todas las anteriores y decide **a cuáles hacer
caso**, en vez de depender de un estado que se arrastra. Eso permite relacionar
cosas lejanas directamente y procesar toda la secuencia en paralelo.

**La matemática.** De cada entrada salen tres proyecciones: consulta $Q$, clave
$K$ y valor $V$.

$$\text{Atención}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $QK^\top$ mide **cuánto se parece** cada consulta a cada clave.
- $\sqrt{d_k}$ evita que el producto crezca con la dimensión y sature el softmax.
  Olvidar esa división es uno de los errores clásicos.
- **Causal**: se pone $-\infty$ en las posiciones futuras antes del softmax, para
  que cada token solo vea el pasado.
- **Multi-cabeza**: varias atenciones en paralelo sobre trozos del vector.

**En el código.** `src/layers/attention.cpp`, con
`float scale = 1.0f / std::sqrt(head_dim_)`. Al lado,
`MultiHeadAttentionReference`, la versión literal para verificar.

**El backward.** El del softmax dentro de la atención es donde más se falla:

$$\frac{\partial L}{\partial s_{ij}} = p_{ij}\left(\frac{\partial L}{\partial p_{ij}} - \sum_k \frac{\partial L}{\partial p_{ik}} p_{ik}\right)$$

Está así, literal, en `attention.cpp`.

**Referencia.** Vaswani et al. (2017), arXiv:1706.03762.

---

## Posición: aprendida y RoPE

**El problema.** La atención no tiene noción de orden: si se barajan los tokens,
el resultado es el mismo barajado. Hay que **meter la posición** de alguna forma.

**Solución clásica.** Sumar un vector aprendido por posición (una tabla `wpe`).
Funciona, pero fija el contexto máximo y depende de posiciones absolutas.

**RoPE.** En vez de sumar, **rotar**. Cada par de canales se rota un ángulo
proporcional a la posición:

$$\theta_i = \frac{\text{pos}}{10000^{2i/d}}$$

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1}\end{pmatrix} =
\begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1}\end{pmatrix}$$

**Por qué es elegante.** El producto escalar entre una consulta rotada y una
clave rotada depende **solo de la diferencia de posiciones**, no de las
absolutas. La atención aprende «lo que está tres tokens atrás», que es lo que se
quería.

Y como la rotación es ortogonal, **el backward es la rotación inversa**: no hay
pesos nuevos ni derivadas complicadas.

**En el código.** `RotarPares` en `src/tensor.cpp`, aplicado en `attention.cpp`.
Rota pares **adyacentes** $(2i, 2i+1)$, la convención del artículo.

**Consecuencia práctica.** Con RoPE el modelo **no tiene tabla de posiciones**,
así que el archivo de pesos lo declara en sus metadatos y `generate_llm`
necesita el mismo `--rope` con el que se entrenó.

**Referencia.** Su et al. (2021), arXiv:2104.09864.

---

## KV-Cache

**La idea.** Al generar token a token, cada paso vuelve a calcular las claves y
valores de todo el texto anterior, que no han cambiado. Guardarlos convierte un
coste cuadrático en lineal.

**El detalle interesante.** Cuando el texto supera la ventana de contexto:

- **con posición aprendida** hay que **reconstruir** la caché, porque cada token
  cambia de índice de posición;
- **con RoPE** basta con **desalojar** el más antiguo, porque solo importan las
  diferencias de posición.

Medido: 1.69 ms/token reconstruyendo, **0.06 ms/token** desalojando.

**En el código.** `MultiHeadAttention::ForwardWithKVCache` y `RecortarKVCache`.

**Verificación.** El Test 38 compara **logits paso a paso** contra la ruta sin
caché. Comparar solo el texto generado no bastaba: el argmax absorbe diferencias
pequeñas, y así estuvo escondido un fallo que reinyectaba un token.

---

## Pesos compartidos

**La idea.** La matriz que convierte tokens en vectores y la que convierte
vectores en predicciones hacen trabajos simétricos. Usar **la misma** ahorra
parámetros y liga ambas representaciones.

**La trampa.** Esa matriz recibe gradiente **por dos caminos**: como embedding de
entrada y como proyección de salida. Hay que **sumarlos**. En este proyecto, el
`Backward` del embedding reiniciaba el acumulador después de que el modelo ya
hubiera sumado su parte, y la contribución de la salida se perdía. El modelo
entrenaba peor sin que nada fallara.

**Verificación.** Test 6, y la paridad contra PyTorch, que detectó el error en
`wte.weight` con una diferencia de 1.0.

**Referencia.** Press y Wolf (2016), arXiv:1608.05859.

---

# Difusión

## El proceso hacia delante

**La idea.** Se define un proceso que **destruye** una imagen añadiéndole ruido
poco a poco, hasta dejar ruido puro. Aprender a **deshacerlo** es aprender a
generar.

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\big)$$

**El atajo que lo hace práctico.** Con $\alpha_t = 1-\beta_t$ y
$\bar\alpha_t = \prod_{s\le t}\alpha_s$, se puede saltar directamente a cualquier
paso:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\;\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

Sin esto habría que simular $t$ pasos por imagen; con esto, uno.

**Por qué esos dos coeficientes.** Están elegidos para que la **varianza se
conserve**: si $x_0$ tiene varianza 1, $x_t$ también, porque
$(\sqrt{\bar\alpha})^2 + (\sqrt{1-\bar\alpha})^2 = 1$.

**En el código.** `DiffusionSchedule::QSample`, con el producto acumulado
calculado en doble precisión.

**El detalle que cuesta caro.** Los valores de $\beta$ del artículo están
calibrados para 1000 pasos. Con 200 pasos y los mismos valores queda **el 36 % de
la imagen** en $x_T$: el modelo nunca ve ruido puro, y al generar arranca de una
distribución que no conoce. No da error, da manchas. Por eso
`DiffusionSchedule::SenalResidual()` existe y el entrenador la imprime.

**Referencia.** Ho, Jain y Abbeel (2020), arXiv:2006.11239.

---

## El objetivo de entrenamiento

**La idea.** En vez de predecir la imagen limpia, se predice **el ruido que se
añadió**. Resulta equivalente y funciona mucho mejor en la práctica.

$$L = \mathbb{E}_{x_0,\varepsilon,t}\left[\;\lVert \varepsilon - \varepsilon_\theta(x_t, t)\rVert^2\;\right]$$

**Lo que enseñó medirlo.** El error de predecir $\varepsilon$ **baja con $t$
grande**, al revés de lo que sugiere la intuición: cuando $\bar\alpha \to 0$,
$\varepsilon \approx x_t$ y la red casi puede copiar su entrada. Lo difícil con
$t$ grande es predecir $x_0$, no $\varepsilon$. Medido: 0.1397 en $t\in[0,50)$
frente a 0.0417 en $t\in[150,200)$.

Consecuencia: **una pérdida baja en la franja alta no dice nada**, porque la
solución trivial ya la consigue.

---

## Muestrear: DDPM y DDIM

**DDPM.** Recorre todos los pasos hacia atrás:

$$\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta\right), \qquad
x_{t-1} = \mu_t + \sigma_t z$$

con la **varianza posterior** $\sigma_t^2 = \beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$, no $\beta_t$ a secas.

**DDIM.** Reescribe el proceso de forma **no markoviana** con las mismas
marginales, y eso permite **saltarse pasos**:

$$x_0^{\text{est}} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}$$
$$x_{\text{prev}} = \sqrt{\bar\alpha_{\text{prev}}}\;x_0^{\text{est}} + \sqrt{1-\bar\alpha_{\text{prev}}-\sigma^2}\;\varepsilon_\theta + \sigma z$$

Con $\eta = 0$ es determinista; con $\eta = 1$ recupera DDPM.

**El fallo que apareció aquí, y por qué es instructivo.** Al recortar
$x_0^{\text{est}}$ a $[-1,1]$ hay que **recalcular** $\varepsilon$ a partir del
$x_0$ recortado. Sin eso el paso es incoherente y, con $\eta = 0$, no hay ruido
que lave el error: **las muestras empeoraban al añadir pasos** (cociente 0.59 con
100 pasos, 0.71 con 500). Corregido, 100 y 500 pasos dan el mismo resultado.

**Cómo se verifica un muestreador.** Con un predictor **perfecto** que siempre
implica la misma imagen, ambos muestreadores deben aterrizar exactamente en ella.
Parece una prueba fortísima y **solo detectaba 1 de 4 fallos**: el predictor
perfecto se corrige solo en cada paso, así que fija el destino pero no el camino.
Hizo falta añadir pruebas sobre la trayectoria.

**Referencias.** Ho et al. (2020); Song, Meng y Ermon (2020), arXiv:2010.02502.

---

## El paso del tiempo, y la U-Net

**El embedding del tiempo.** La red tiene que saber **en qué paso está**, porque
quitar ruido al principio y al final son tareas distintas. Pasarle el número
crudo no sirve: se codifica con senos y cosenos de distintas frecuencias, igual
que la posición en el Transformer.

$$\text{emb}(t) = [\sin(t\omega_1),\dots,\sin(t\omega_{d/2}),\cos(t\omega_1),\dots,\cos(t\omega_{d/2})]$$

**En el código.** `diffusion/time_embedding.h`. El tiempo entra en cada bloque
residual **sumado por canal**: un valor por canal aplicado a todo su mapa, porque
el paso es una propiedad de la imagen entera.

**La U-Net.** Baja de resolución para ver la forma global y sube para recuperar
el detalle, con **saltos** que llevan los mapas de la bajada al nivel
correspondiente de la subida.

**El detalle del backward.** Cada salto recibe gradiente **por dos caminos** —el
de la bajada y el de la concatenación— y hay que **sumarlos**. Sobrescribir en
vez de sumar es el error clásico; aquí lo detecta una prueba de diferencias
finitas de extremo a extremo.

**Referencia.** Ronneberger et al. (2015), arXiv:1505.04597.

---

# Difusión latente

## Autoencoder variacional

**La idea.** Comprimir la imagen a un espacio pequeño **y ordenado**. Un
autoencoder normal comprime, pero su espacio latente puede tener huecos y saltos;
si se difunde sobre él, los puntos intermedios no corresponden a imágenes.

**La solución.** Que el codificador no produzca un punto sino **una distribución**
—una media y una varianza— y que el latente se sortee de ella. Eso obliga a que
puntos cercanos den imágenes parecidas.

**Reparametrización.** Sortear directamente cortaría el gradiente. Se escribe el
azar como una entrada más:

$$z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

Así $\partial z/\partial \mu$ y $\partial z/\partial \sigma$ existen y el
gradiente atraviesa el sorteo.

**La penalización KL.** Mantiene la distribución cerca de una gaussiana estándar:

$$D_{KL} = \tfrac{1}{2}\sum \left(\mu^2 + e^{\log\sigma^2} - 1 - \log\sigma^2\right)$$

Vale **exactamente 0** cuando $\mu = 0$ y $\log\sigma^2 = 0$, lo que da un caso
de prueba con respuesta conocida.

**Los tres errores típicos**, los tres cubiertos por mutaciones:

1. Olvidar el $\tfrac{1}{2}$ al derivar $\sigma = e^{\log\sigma^2/2}$.
2. No dividir la KL por el tamaño del lote.
3. Dejar pasar gradiente por donde el recorte de $\log\sigma^2$ actuó.

**En el código.** `latent/gaussiana.h`. `Backward(dz, peso_kl)` suma dentro los
dos caminos —reconstrucción y KL— para que ningún entrenador olvide uno.

**Referencia.** Kingma y Welling (2013), arXiv:1312.6114.

---

## Difundir en el latente

**La idea del artículo.** Difundir sobre píxeles es caro porque el coste crece
con el número de posiciones. Si primero se comprime la imagen por un factor $f$,
la difusión trabaja sobre $f^2$ veces menos posiciones.

**Medido aquí**, con la misma U-Net: 40 ms por paso sobre un latente 8×8 frente a
252 ms sobre píxeles 32×32. **6.3 veces más barato.**

**La escala del latente.** La difusión supone datos centrados y de varianza
cercana a uno —su calendario está calibrado para eso— y el latente de un
autoencoder con KL débil no tiene por qué cumplirlo. Se mide y se aplica:

$$z' = (z - \text{media}) \times \text{escala}, \qquad \text{escala} = 1/\sigma$$

**Diferencia con el artículo:** allí basta con dividir por $\sigma$ porque sus
latentes ya salen centrados. Aquí la media resultó ser $-0.69$ con $\sigma =
0.61$, más de una desviación, así que **se sellan media y escala**.

**Referencia.** Rombach et al. (2021), arXiv:2112.10752.

---

# Optimización

## Adam y AdamW

**La idea.** El descenso por gradiente simple usa el mismo paso para todos los
pesos. Adam adapta el paso de cada uno según su historial: mantiene dos medias
móviles, del gradiente y de su cuadrado.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
\theta \mathrel{-}= \frac{\eta\,\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**La corrección de sesgo** ($1-\beta^t$) compensa que $m$ y $v$ arranquen en cero.
Tiene una consecuencia práctica: **los primeros pasos son del tamaño máximo**,
justo cuando los pesos son aleatorios. Por eso se usa calentamiento.

**AdamW.** Aplica el decaimiento de pesos **por separado** del gradiente, en vez
de sumarlo a él. Con Adam clásico, el decaimiento se dividía por $\sqrt{\hat v}$
y acababa siendo distinto para cada peso.

**Consecuencia para reanudar.** El estado de Adam **es parte del modelo**:
reanudar con $m = v = 0$ da una sacudida. Por eso el checkpoint los guarda.

**Referencias.** Kingma y Ba (2014); Loshchilov y Hutter (2017).

### El punto de partida: SGD con momento

Antes que Adam, y todavía en `optimizers.h`:

$$v_t = \mu\,v_{t-1} + g_t, \qquad \theta \mathrel{-}= \eta\,v_t$$

El **momento** $\mu$ acumula la dirección de los pasos anteriores: amortigua el
zigzagueo cuando el gradiente cambia de signo y acelera cuando es consistente.
Adam puede verse como esta idea más una escala por peso.

---

## Media móvil de pesos (EMA)

**La idea.** El gradiente de difusión es muy ruidoso —cada paso ve un $t$
sorteado y un ruido nuevo—, así que los pesos finales oscilan alrededor del
bueno. Promediarlos cancela esa oscilación.

$$\bar\theta_t = d\,\bar\theta_{t-1} + (1-d)\,\theta_t$$

**El detalle práctico.** Con $d$ fijo, la sombra arrastra los **pesos iniciales
al azar** durante $\approx 1/(1-d)$ pasos. Por eso el decaimiento efectivo sube
desde cero: $\min(d, (1+n)/(10+n))$.

**Y su contador es estado.** Al reanudar hay que restaurarlo, o la rampa se
reinicia y la sombra se reengancha a los pesos vivos. **Ninguna pérdida lo
detecta**, porque la pérdida usa los pesos vivos: se encontró comparando
archivos byte a byte.

**Referencia.** Polyak y Juditsky (1992); su uso en difusión, Ho et al. (2020).

---

## Tasa de aprendizaje

$$\eta_t = \begin{cases}
\eta \cdot t/T_w & t \le T_w \quad \text{(calentamiento)}\\[4pt]
\eta_{\min} + \tfrac{1}{2}(\eta - \eta_{\min})\left(1 + \cos\pi\frac{t - T_w}{T - T_w}\right) & t > T_w
\end{cases}$$

**Consecuencia práctica que se midió.** Como el coseno se calcula sobre el total
$T$, **cambiar `--iteraciones` cambia el entrenamiento**, no solo su duración.
Por eso existe `--parar_en`, que corta sin alterar el plan, y por eso el total va
sellado en el checkpoint.

**Referencia.** Loshchilov y Hutter (2016), *SGDR*, arXiv:1608.03983.

---

## Por qué el resultado es reproducible

Dos decisiones que hacen posible todo lo demás:

1. **El reparto entre hilos no cambia el resultado.** Cada hilo escribe
   posiciones que ningún otro toca: no hay reducción, y sumar en otro orden daría
   otros últimos decimales. Con eso, 1 hilo y 10 hilos dan **exactamente** lo
   mismo, y la comparación contra PyTorch tiene sentido.

2. **Cada iteración se siembra con `(semilla, iteración)`**, en vez de arrastrar
   el estado del generador. Así la iteración *n* usa el mismo lote y el mismo
   ruido tanto si se llega de un tirón como reanudando.

Verificado con el modelo real: reanudar desde la iteración 70 000 hasta la 80 000
dio pesos, EMA y momentos de Adam **idénticos bit a bit**.

---

## Para seguir

- **Cómo se verifica cada cosa**: [VERIFICACION.md](VERIFICACION.md)
- **Dónde está cada cosa**: [MAPA_DEL_CODIGO.md](MAPA_DEL_CODIGO.md)
- **Cómo encaja todo**: [ARQUITECTURA.md](ARQUITECTURA.md)
- **La bibliografía completa**: [REFERENCIAS.md](REFERENCIAS.md)
- **Derivaciones largas del Transformer**: [DOCS_MATHEMATICS.md](../DOCS_MATHEMATICS.md)
