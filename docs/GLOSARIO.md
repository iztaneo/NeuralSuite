# Glosario

Los términos que aparecen en el código, los documentos y los logs, explicados
una vez y sin dar por sabido nada. Ordenados por tema, no alfabéticamente,
porque cada uno se entiende mejor junto a los suyos.

Cada entrada cabe en cuatro líneas a propósito: esto es para tener abierto al
lado del código, no para estudiarlo. La explicación larga de cada concepto —con
su matemática y su verificación— está en
[TEORIA_E_IMPLEMENTACION.md](TEORIA_E_IMPLEMENTACION.md), y de qué artículo sale,
en [REFERENCIAS.md](REFERENCIAS.md).

Cuando un término describe algo que **no está implementado aquí**, se dice. Lo
que hay y lo que no, pieza a pieza, está en [ESTADO.md](ESTADO.md).

---

## 1. Matemática y tensores

**Tensor.** Una matriz de números con forma. `[32, 1, 28, 28]` son 32 imágenes de
un canal y 28×28 píxeles. Es el tipo sobre el que gira todo.

**Forma (*shape*).** La lista de tamaños de cada dimensión. Casi todos los
defectos de este proyecto empezaron por una forma que no era la esperada, y por
eso `Tensor` la valida y los mensajes de error dicen qué llegó y qué se esperaba.

**Dimensión o eje.** Cada una de las posiciones de la forma. En `[B, C, H, W]`,
el eje 0 es el lote, el 1 los canales, y el 2 y 3 alto y ancho.

**Rango (*rank*).** Cuántas dimensiones tiene un tensor: un vector tiene rango 1,
una imagen con canales rango 3, un lote de imágenes rango 4.

**Almacenamiento (*storage*).** El bloque de memoria plano donde viven los
números, separado de la forma que se les da. Aquí es un `std::vector<float>` con
propiedad compartida.

**Vista (*view*).** Otro tensor con **otra forma sobre la misma memoria**, sin
copiar nada. Aplanar `[B, T, C]` a `[B·T, C]` antes de una capa densa es gratis.
Cuidado: **copiar un `Tensor` copia los datos**; solo `View()` los comparte.

**Orden de memoria.** Los tensores del proyecto son **contiguos** en orden por
filas: el último eje es el que varía más rápido. No hay `stride` arbitrario, lo
que simplifica el código a cambio de obligar a copiar en algunas operaciones.

**Transposición.** Intercambiar dos ejes. En la atención se usa constantemente
para alinear `[lote, cabeza, paso, canal]` con la multiplicación que toca.

**MatMul y GEMM.** Multiplicación de matrices. *GEMM* es su nombre en las
bibliotecas de álgebra lineal, y aquí está escrita a mano en `src/tensor.cpp`:
es la operación que se lleva la mayor parte del tiempo de cálculo.

**Reducción.** Colapsar un eje sumando, promediando o tomando el máximo. Es la
operación que **no se puede repartir entre hilos sin cambiar el resultado**, y de
ahí que `parallel.h` la evite entre hilos.

**Difusión de formas (*broadcasting*).** Operar dos tensores de formas distintas
estirando virtualmente el pequeño, como sumar un sesgo `[C]` a un `[B, C]`. En
NeuralSuite solo lo hace el motor de autograd; las capas escritas a mano tratan
cada caso explícitamente.

**Norma.** La longitud de un vector, `√Σxᵢ²`. Aparece al medir gradientes y al
recortarlos.

**Épsilon (`eps`).** Un número minúsculo que se suma a un denominador para no
dividir entre cero. En las normalizaciones, `1e-5`.

---

## 2. Entrenamiento y optimización

**Parámetro (peso).** Un número que el modelo aprende. Un modelo «de 858 000
parámetros» tiene esa cantidad de números ajustables.

**Hiperparámetro.** Un número que **tú** eliges y el modelo no aprende: tasa de
aprendizaje, tamaño del lote, número de capas. Suelen ser las opciones de línea
de órdenes de los programas de `apps/`.

**Gradiente.** Para cada peso, en qué dirección y cuánto habría que moverlo para
que el error baje. Es la derivada del error respecto a ese peso.

**Forward y backward.** *Forward* es calcular la salida a partir de la entrada;
*backward* es propagar el error hacia atrás para obtener los gradientes.

**Pérdida (*loss*).** El número que mide lo mal que lo está haciendo el modelo, y
que el entrenamiento intenta minimizar. También se le llama función objetivo.

**Logits.** Las puntuaciones que salen de la última capa, **antes** de
convertirlas en probabilidades. Pueden ser negativas y no suman uno.

**Softmax.** Convierte logits en probabilidades: exponencia y normaliza. Se
calcula restando el máximo primero, porque si no, `exp` desborda.

**Entropía cruzada.** La pérdida de los problemas de clasificación y de los
modelos de lenguaje: mide cuánta probabilidad le dio el modelo a la respuesta
correcta. Sale de la máxima verosimilitud.

**MSE (error cuadrático medio).** La pérdida de los problemas de regresión: el
promedio de `(predicho − real)²`. Es la del autoencoder y la de la difusión.

**Lote (*batch*).** Cuántos ejemplos se procesan a la vez. Lotes grandes dan
gradientes menos ruidosos y ocupan más memoria.

**Iteración y época.** Una *iteración* es un lote. Una *época* es haber pasado
una vez por todo el conjunto de datos.

**Optimizador.** Lo que decide cómo mover los pesos dado el gradiente. Aquí:
`SGD`, `Adam` y `AdamW`.

**SGD (descenso por gradiente estocástico).** El más simple: restar el gradiente
multiplicado por la tasa de aprendizaje.

**Momento (*momentum*).** Arrastrar parte del paso anterior, como una bola que
baja una pendiente. Suaviza el ruido y atraviesa zonas planas.

**Adam.** Mantiene dos medias móviles por peso —del gradiente y de su cuadrado— y
adapta el tamaño del paso de cada uno. Es el que usan todos los entrenamientos
del proyecto.

**AdamW.** Adam con el **decaimiento de pesos desacoplado**: en vez de meterlo en
el gradiente, se resta aparte. La diferencia importa, y es un artículo entero.

**Decaimiento de pesos (*weight decay*).** Empujar todos los pesos ligeramente
hacia cero en cada paso, para que el modelo no se apoye demasiado en ninguno.

**Tasa de aprendizaje (*learning rate*).** Cuánto se mueven los pesos en cada
paso. Demasiado alta y el entrenamiento se desestabiliza; demasiado baja y no
avanza.

**Calentamiento (*warmup*) y coseno.** Subir la tasa poco a poco al principio y
bajarla suavemente hasta el final. El calentamiento evita que los primeros pasos,
con pesos aleatorios, sean enormes.

**Recorte de gradiente (*gradient clipping*).** Si la norma del gradiente pasa de
un umbral, se escala entera hacia abajo. Evita que un lote raro reviente el
entrenamiento; `train_llm` recorta a 1.0.

**Inicialización Xavier.** Sortear los pesos iniciales con una varianza que
depende del tamaño de la capa, para que la señal no se apague ni explote al
atravesar muchas capas.

**EMA (media móvil de pesos).** Guardar un promedio suavizado de los pesos
durante el entrenamiento. En difusión se muestrea con él, porque los pesos
finales oscilan alrededor del bueno.

**Acumular gradiente.** Que `Backward` **sume** sobre lo que ya había en vez de
sobrescribirlo. Es lo que permite que un peso reciba gradiente por dos caminos,
como en un bloque residual.

**Poner a cero (`ZeroGrad`).** Vaciar los gradientes antes de cada iteración.
Olvidarlo es el error clásico: los gradientes de los lotes se suman.

---

## 3. Grafo de cómputo y autograd

**Grafo de cómputo.** La representación de qué operaciones transformaron las
entradas en la salida. Cada nodo sabe qué operación hizo y cómo propagar un
gradiente hacia sus entradas.

**Autograd (diferenciación automática).** Un sistema que construye ese grafo
durante el *forward* y aplica la regla de la cadena automáticamente en el
*backward*. NeuralSuite tiene uno, en `include/autograd.h`, con 12 primitivas.

**Backward manual.** La alternativa que usan **todas las capas del proyecto**:
cada capa deriva su propio *forward* a mano y guarda lo que necesitará. Es más
rápido y aquí no encontró ni un defecto menos; el porqué de esa decisión está
medido en [AUTOGRAD_CAPAS.md](AUTOGRAD_CAPAS.md).

**Regla de la cadena.** La regla de derivación en la que se apoya todo:
la derivada de una composición es el producto de las derivadas. El *backward* de
una red es exactamente eso, aplicado capa a capa desde el final.

**Diferencias finitas.** Comprobar un gradiente moviendo un peso un poquito
arriba y abajo y midiendo cómo cambia el error. Verifica que el *backward* es la
derivada del *forward*.

---

## 4. Capas y redes

**Capa densa (`Linear`).** Multiplica por una matriz y suma un vector. La
operación básica.

**Activación.** Una función no lineal entre capas: sin ella, apilar capas sería
equivalente a una sola.

**ReLU.** `max(0, x)`. La más simple y la más usada; su derivada es 0 o 1.

**GELU.** Una versión suave de ReLU que no corta de golpe en cero. Es la del MLP
del GPT.

**SiLU (*swish*).** `x · sigmoide(x)`. La activación de la difusión y del
autoencoder.

**Sigmoide y tanh.** Funciones que aplastan cualquier número a `(0,1)` y `(−1,1)`
respectivamente. Son las puertas de la LSTM.

**Normalización.** Reescalar valores intermedios para que el entrenamiento sea
estable. `LayerNorm` normaliza cada ejemplo; `GroupNorm`, por grupos de canales;
`RMSNorm` es una versión más barata.

**RMSNorm.** Normaliza solo por la raíz de la media de los cuadrados, sin restar
la media. Verificada contra PyTorch, pero **el GPT de este proyecto no la usa**.

**Bloque residual.** Una rama que calcula algo y se **suma** a la entrada en vez
de reemplazarla. El «atajo» deja pasar el gradiente intacto hacia abajo.

**Recurrente (`LSTM`).** Procesa una secuencia paso a paso arrastrando un estado.
`BiLSTM` la recorre en los dos sentidos.

**Puerta (*gate*).** En una LSTM, un vector entre 0 y 1 que decide cuánto dejar
pasar: olvidar, entrar o salir. El orden de las puertas importa, y es donde la
paridad con PyTorch encuentra los desajustes.

**BPTT.** *Retropropagación a través del tiempo*: el *backward* de una
recurrente, que recorre la secuencia hacia atrás acumulando gradiente en los
mismos pesos en cada paso.

**Módulo.** Una pieza que declara sus parámetros y puede contener otras. De ahí
salen los nombres con ruta —`blocks.0.attn.weight`— que usa el formato de pesos.

**Dropout.** Apagar neuronas al azar durante el entrenamiento para que la red no
dependa de ninguna. **No está implementado**: ningún modelo del proyecto lo ha
necesitado, y en la paridad se fuerza a cero en el lado de PyTorch.

---

## 5. Transformer y lenguaje

**Token.** La unidad mínima de texto. Aquí es **un carácter**; en modelos grandes
suele ser un fragmento de palabra.

**Vocabulario.** El conjunto de tokens que el modelo conoce. El de `es_base` son
113 caracteres más `<UNK>`.

**BPE (*byte pair encoding*).** Construir el vocabulario fusionando los pares de
caracteres más frecuentes, para que una palabra común sea un solo token.
**Pendiente**: acortaría mucho las secuencias.

**Embedding.** Tabla que convierte un token (un número) en un vector.

**Contexto (`block_size`).** Cuántos tokens puede mirar el modelo a la vez. En
`es_base`, 128 caracteres.

**Atención.** Cada posición mira a las demás y decide a cuáles hacer caso. Es lo
que reemplazó a las recurrentes en el procesamiento de lenguaje.

**Query, key y value.** Los tres vectores que se calculan de cada posición: la
*query* es lo que esa posición busca, la *key* lo que ofrece, y el *value* lo que
aporta si la eligen. La atención empareja queries con keys y mezcla values.

**Producto escalar escalado (*scaled dot-product attention*).** La fórmula de la
atención: emparejar cada query con cada key por producto escalar, **dividir por
`√d`**, pasar por softmax y usar el resultado para promediar los values. La
división evita que los logits crezcan con la dimensión y saturen el softmax.

**Cabeza (*head*) y dimensión de cabeza.** La atención se parte en varias
cabezas paralelas, cada una con una fracción de los canales —`d_head = n_embd /
n_head`—, y sus salidas se concatenan.

**Multi-cabeza.** Varias atenciones en paralelo, cada una atenta a algo distinto.

**Autoatención (*self-attention*).** Cuando queries, keys y values salen de la
**misma** secuencia. Es la del GPT.

**Atención cruzada (*cross-attention*).** Cuando las queries salen de una
secuencia y las keys y values de **otra**: el mecanismo con el que un texto
condicionaría una imagen. Implementada y verificada, pero **ningún modelo la usa
todavía**.

**Máscara causal.** Poner a `−∞` los logits de atención hacia el futuro, antes
del softmax, para que cada posición solo mire hacia atrás. Es lo que permite
entrenar todas las posiciones a la vez y aun así generar palabra a palabra.

**Codificador y decodificador.** En el Transformer original, un codificador lee
la entrada entera y un decodificador produce la salida. Un modelo
**decodificador solo** —como el GPT— se salta el primero y genera directamente.

**MLP del bloque (*feed-forward*).** Las dos capas densas con una activación en
medio que van después de la atención en cada bloque, y que expanden a `4×` los
canales antes de volver.

**Pre-LN y post-LN.** Dónde se normaliza: antes de cada subcapa o después de
sumar el residual. **Pre-LN** —lo que hace este GPT— entrena de forma más estable
sin calentamiento largo.

**SwiGLU.** Un MLP de bloque con una puerta multiplicativa en vez de una
activación simple. Verificado contra PyTorch, **no integrado en el GPT**.

**Posición aprendida.** Una tabla de vectores, uno por posición, que se suma al
embedding. Es lo que usa `es_base`.

**RoPE (posición rotatoria).** En vez de sumar una posición, **rota** los pares
de canales de queries y keys según su posición. El efecto es que el producto
entre una query y una key incorpora de forma natural la posición **relativa**
entre ambas.

**KV-Cache.** Al generar, guardar las claves y valores ya calculados para no
recalcular todo el texto anterior en cada token nuevo.

**Ventana deslizante.** Cuando el caché llega a su límite, desalojar lo más
antiguo en vez de reconstruirlo todo. Con RoPE eso es correcto porque la posición
va dentro del vector.

**GQA (*grouped-query attention*).** Compartir keys y values entre varias
cabezas para ahorrar memoria de caché. **Descartado con medición**: con solo 4
cabezas ahorraba 0.79 MB y recortaba capacidad.

**Pesos compartidos (*weight tying*).** Usar la misma matriz para convertir
tokens en vectores al entrar y para producir la predicción al salir.

**Temperatura.** Al generar, cuánto azar se permite. Baja: repetitivo y seguro.
Alta: variado y con más errores.

**Muestreo autorregresivo.** Generar un token, añadirlo a la entrada y repetir.
Es la razón de que exista el KV-Cache.

---

## 6. Visión y convolución

**Convolución (`Conv2D`).** Desliza un filtro pequeño por toda la imagen. Sirve
para imágenes porque busca el mismo patrón en cualquier posición.

**Canal.** Cada uno de los «planos» de una imagen o de un mapa de
características: 1 en escala de grises, 3 en color, 32 dentro de la U-Net.

**Núcleo (*kernel*) y su tamaño.** El filtro que se desliza. Aquí casi siempre
3×3, y 1×1 cuando solo hay que cambiar el número de canales.

**Relleno (*padding*) y paso (*stride*).** Cuántos ceros se añaden alrededor y
de cuánto en cuánto se desliza el filtro. Con 3×3, relleno 1 y paso 1, la salida
conserva el tamaño.

**Campo receptivo.** Cuánta imagen original influye en un valor de salida. Crece
al apilar convoluciones, y es la razón de bajar de resolución.

**`im2col`.** Reorganizar los parches de la imagen como columnas de una matriz
para convertir la convolución en una multiplicación de matrices. Es lo que hizo
a `Conv2D` 55 veces más rápida.

**Submuestreo (`MaxPool2D`, `Downsample2D`).** Reducir alto y ancho a la mitad,
tomando el máximo de cada ventana o convolucionando con paso 2.

**Ampliación (`Upsample2D`).** Duplicar alto y ancho. Aquí se hace **repitiendo
el vecino más próximo y convolucionando después**, nunca con convolución
transpuesta, para evitar el patrón de tablero.

**Patrón de tablero (*checkerboard*).** El artefacto cuadriculado que deja la
convolución transpuesta cuando el paso no divide al núcleo.

**CRNN.** La arquitectura del OCR: convoluciones que leen la imagen, una
recurrente que recorre las columnas y una salida por columna.

**CTC.** La pérdida estándar para leer texto **sin saber dónde cae cada letra**.
**No está implementada**: el corpus de OCR se genera con la alineación conocida,
así que se entrena con entropía cruzada por columna.

---

## 7. Difusión

**Difusión.** Entrenar una red para **quitar ruido**, y luego generar partiendo
de ruido puro y quitándoselo poco a poco.

**Proceso directo (*forward*).** Añadir ruido a una imagen limpia. Tiene fórmula
cerrada, así que se puede saltar a cualquier `t` de una vez, sin simular los
pasos intermedios.

**Proceso inverso.** Quitar ruido paso a paso partiendo de ruido puro. Es lo que
hace un muestreador, y lo que la red aprende a guiar.

**`x₀`.** La imagen limpia original.

**`x_t`.** La imagen después de añadirle ruido hasta el instante `t`.

**`t` (paso de ruido).** Cuánto ruido lleva una imagen: `t = 0` es la imagen
limpia y `t = T` es ruido puro. Se le pasa a la red como entrada.

**`ε` (épsilon).** El ruido gaussiano que se añadió. **Es lo que la red predice**:
el objetivo del entrenamiento es adivinar qué ruido lleva `x_t`.

**`β_t` (beta).** Cuánto ruido se añade en el paso `t`. Crece con `t`, de
`1e-4` a `0.02` en el calendario lineal de 1000 pasos.

**`α_t` y `ᾱ_t` (alfa y alfa barra).** `α = 1 − β` es la señal que sobrevive a un
paso; `ᾱ` es su producto acumulado, la señal que queda después de `t` pasos. La
fórmula entera es `x_t = √ᾱ·x₀ + √(1−ᾱ)·ε`.

**Calendario de ruido (*noise schedule*).** La receta completa de `β` para todos
los pasos. Está calibrada para 1000 pasos: con menos, hay que reescalarla o la
imagen final no llega a ser ruido.

**Desruidificador (*denoiser*).** La red que, dado `x_t` y `t`, predice el ruido.
En este proyecto es una `UNet2D`.

**U-Net.** La red de la difusión: baja de resolución para ver la forma global,
vuelve a subir, y usa **saltos** para no perder el detalle fino.

**Conexión de salto (*skip connection*).** El puente que lleva el mapa de la
bajada a la subida correspondiente, para recuperar el detalle que el
submuestreo perdió.

**Embedding del tiempo.** Cómo se le dice a la red en qué `t` está: se convierte
el entero en un vector con senos y cosenos de varias frecuencias, y se inyecta en
cada bloque.

**Posterior.** La distribución de `x_{t−1}` dados `x_t` y `x₀`. Su media y su
**varianza** son lo que usa el muestreador; usar `β` en lugar de esa varianza es
un error común, y aquí se usa la correcta.

**Muestreador (*sampler*).** El algoritmo que recorre la trayectoria inversa.
Recibe la red como una función `(x_t, t) → ε`, lo que permite probarlo contra un
oráculo analítico sin modelo entrenado.

**DDPM.** El muestreador original: recorre todos los pasos, con algo de azar en
cada uno.

**DDIM.** Un muestreador que puede **saltarse pasos** y, con `eta = 0`, es
determinista: la misma semilla da siempre la misma imagen.

**`eta`.** Cuánto azar se le devuelve a DDIM. Con `0` es determinista; con `1`
recupera el comportamiento de DDPM.

**Recorte de `x₀` (*clipping*).** Forzar la imagen estimada al rango válido en
cada paso. Si se hace, **hay que recalcular `ε` para que sea coherente**: no
hacerlo empeoraba las muestras al añadir pasos.

**Guía sin clasificador (*classifier-free guidance*).** Amplificar la diferencia
entre la predicción condicionada y la incondicional para que la muestra se
parezca más a lo pedido. **No aplica todavía**: este modelo no está
condicionado.

---

## 8. Autoencoders y espacio latente

**Autoencoder.** Un par de redes: el **codificador** comprime y el
**decodificador** reconstruye.

**Espacio latente.** El espacio donde viven las versiones comprimidas. Aquí un
latente es `8×8×4` números en vez de los `32×32` píxeles.

**Factor de compresión.** Cuántas veces menos números tiene el latente. Con
`C = 4`, cuatro veces; con `C = 1`, dieciséis.

**Cuello de botella.** La parte más estrecha del autoencoder, que obliga a
quedarse con lo esencial. Si no fuera estrecha, copiar sería una solución válida.

**Pérdida de reconstrucción.** Lo que mide cuánto se parece la salida a la
entrada. Aquí, error cuadrático medio.

**Pérdida perceptual.** Comparar imágenes por las características de una red
preentrenada en vez de píxel a píxel. **No implementada**: necesitaría una VGG
ajena, y eso rompería la premisa del proyecto.

**VAE (autoencoder variacional).** Un autoencoder que produce una
**distribución** por cada entrada en vez de un punto, lo que ordena el espacio
latente y permite muestrear de él.

**`μ` y `log σ²` (media y log-varianza).** Los dos vectores que produce el
codificador variacional. Se guarda el logaritmo de la varianza porque así puede
ser negativo sin problemas y es más estable; aquí se acota a `[−30, 20]`.

**Reparametrización.** Truco para poder entrenar cuando hay azar de por medio:
en vez de sortear el latente directamente, se escribe como `media + desviación ×
ruido`, y así el gradiente puede atravesarlo.

**Prior y posterior.** El *prior* es la distribución que **se quiere** en el
latente —una gaussiana estándar—; el *posterior* es la que el codificador
produce para una entrada concreta. La KL mide cuánto se alejan.

**KL (divergencia de Kullback-Leibler).** Medida de cuánto se parece una
distribución a otra. Aquí penaliza que el latente se aleje del prior, lo que
mantiene el espacio ordenado.

**Escala y media del latente (*latent normalization*).** Los dos números que se
aplican antes de difundir —`(z − media) × escala`— porque la difusión supone
datos centrados y de varianza cercana a uno. Se **sellan** en el checkpoint del
autoencoder.

**LDM (modelo de difusión latente).** Difundir en el espacio latente en vez de
en los píxeles. Aquí el autoencoder está hecho y medido (LDM-1); generar sobre su
latente es la Fase 19 (LDM-2).

---

## 9. Evaluación

**Entrenamiento, validación y prueba.** Tres conjuntos separados: con el primero
se aprende, con el segundo se vigila el progreso, y con el tercero —que no se
mira hasta el final— se comprueba si generaliza.

**Generalizar.** Funcionar sobre datos que no se vieron al entrenar. Es lo único
que se mide de verdad; lo demás es descripción del conjunto de entrenamiento.

**Sobreajuste (*overfitting*).** Cuando el modelo memoriza los datos de
entrenamiento en vez de aprender la regla: el error baja en entrenamiento y sube
en validación.

**Brecha de validación.** La diferencia entre la pérdida de entrenamiento y la de
validación. Que se mantenga plana es señal de que no hay sobreajuste **según esa
métrica**, que no es lo mismo que demostrar que no memoriza nada.

**Puerta de sobreajuste.** Aquí, lo contrario y a propósito: pedirle al modelo
que memorice unas pocas imágenes **para comprobar que puede**. Si no lo
consigue, hay un defecto y no vale la pena entrenar horas.

**Perplejidad.** Medida de un modelo de lenguaje: la exponencial de la pérdida.
Se interpreta como una **incertidumbre efectiva**: una perplejidad de 6.23
equivale a dudar como si eligiera entre unas 6.23 alternativas equiprobables, que
no es lo mismo que decir que hay seis candidatos reales.

**PSNR.** Medida de parecido entre dos imágenes, en decibelios, derivada del
error cuadrático. Más alto es menos error, pero **no es una escala absoluta de
calidad percibida**: depende del contenido y del tipo de distorsión, así que se
lee comparando contra un control y mirando las imágenes.

**FID.** Medida de la calidad de imágenes **generadas**: compara la distribución
de las generadas con la de las reales usando las características de un
clasificador. Pendiente, Fase 19.

**Control.** Un punto de comparación que dice si el resultado significa algo: una
red sin entrenar, un codificador congelado, o un compresor trivial del mismo
tamaño.

**Distancia a la vecina más cercana.** Para comprobar que una imagen generada no
es una copia: se busca la imagen de entrenamiento más parecida y se compara esa
distancia con la distancia media.

---

## 10. Ingeniería y reproducibilidad

**Paridad.** Comparar nuestros números con los de PyTorch usando los mismos
pesos y las mismas entradas. Verifica que el *forward* es la arquitectura que se
pretendía, cosa que las diferencias finitas **no pueden ver**.

**Oráculo.** Una fuente externa que sabe la respuesta correcta: PyTorch para las
capas, Pillow para las imágenes, o una fórmula cerrada para los muestreadores.

**Implementación de referencia.** La versión lenta y obvia de una operación
—`Conv2DReference`, `LSTMReference`— que existe solo para comprobar que la rápida
calcula lo mismo.

**Mutación.** Introducir un fallo a propósito para comprobar que la prueba lo
detecta. Una prueba que nunca falla no demuestra nada.

**Checkpoint.** Una foto del entrenamiento: pesos, estado del optimizador y
configuración, con lo necesario para continuar exactamente donde se dejó.

**Sello.** Los metadatos que guarda un checkpoint con toda la configuración del
run. Al reanudar, manda el sello: si la línea de órdenes lo contradice, se
aborta.

**Transaccional.** Escribir todos los archivos de un checkpoint como temporales
y moverlos solo cuando todos están completos, para que un corte no deje una
mezcla.

**NSF.** El formato de archivo del proyecto: número mágico, versión, metadatos,
tensores con nombre y forma, y suma de comprobación.

**Semilla (*seed*).** El número que fija la secuencia de azar. Aquí cada
iteración deriva la suya de la semilla global y del número de iteración, y por
eso reanudar no necesita guardar el estado del generador.

**Determinista / idéntico bit a bit.** Que dos ejecuciones den exactamente los
mismos números, hasta el último decimal. Es lo que permite comparar y verificar.

**Paralelismo sin reducción.** Repartir el trabajo de forma que cada hilo escriba
posiciones que ningún otro toca. Sin sumas en orden variable, el resultado con 10
hilos es idéntico al de 1.

**Procedencia.** Los datos que hacen recuperable un resultado: commit, hash del
conjunto de datos, semilla, build y máquina. Están en [MODELOS.md](MODELOS.md).
