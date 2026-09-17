# Glosario

Los términos que aparecen en el código, los documentos y los logs, explicados
una vez y sin dar por sabido nada. Ordenados por tema, no alfabéticamente,
porque cada uno se entiende mejor junto a los suyos.

Para saber de qué artículo sale cada concepto, ver [REFERENCIAS.md](REFERENCIAS.md);
para cómo está implementado, [TEORIA_E_IMPLEMENTACION.md](TEORIA_E_IMPLEMENTACION.md).

---

## Lo básico

**Tensor.** Una matriz de números con forma. `[32, 1, 28, 28]` son 32 imágenes de
un canal y 28×28 píxeles. Es el tipo sobre el que gira todo.

**Parámetro (peso).** Un número que el modelo aprende. Un modelo «de 858 000
parámetros» tiene esa cantidad de números ajustables.

**Gradiente.** Para cada peso, en qué dirección y cuánto habría que moverlo para
que el error baje. Es la derivada del error respecto a ese peso.

**Forward y backward.** *Forward* es calcular la salida a partir de la entrada;
*backward* es propagar el error hacia atrás para obtener los gradientes.

**Lote (*batch*).** Cuántos ejemplos se procesan a la vez. Lotes grandes dan
gradientes menos ruidosos y ocupan más memoria.

**Iteración y época.** Una *iteración* es un lote. Una *época* es haber pasado
una vez por todo el conjunto de datos.

**Tasa de aprendizaje (*learning rate*).** Cuánto se mueven los pesos en cada
paso. Demasiado alta y el entrenamiento se desestabiliza; demasiado baja y no
avanza.

**Calentamiento (*warmup*) y coseno.** Subir la tasa poco a poco al principio y
bajarla suavemente hasta el final. El calentamiento evita que los primeros pasos,
con pesos aleatorios, sean enormes.

---

## Entrenar y medir

**Entrenamiento, validación y prueba.** Tres conjuntos separados: con el primero
se aprende, con el segundo se vigila el progreso, y con el tercero —que no se
mira hasta el final— se comprueba si generaliza.

**Sobreajuste (*overfitting*).** Cuando el modelo memoriza los datos de
entrenamiento en vez de aprender la regla: el error baja en entrenamiento y sube
en validación.

**Puerta de sobreajuste.** Aquí, lo contrario y a propósito: pedirle al modelo
que memorice unas pocas imágenes **para comprobar que puede**. Si no lo
consigue, hay un defecto y no vale la pena entrenar horas.

**Perplejidad.** Medida de un modelo de lenguaje: aproximadamente, entre cuántas
opciones duda en cada carácter. Más baja es mejor. Perplejidad 6.23 significa que
duda como si eligiera entre unas seis posibilidades.

**PSNR.** Medida de parecido entre dos imágenes, en decibelios. Más alto es
mejor; por encima de 30 dB la diferencia apenas se ve.

**FID.** Medida de la calidad de imágenes **generadas**: compara la distribución
de las generadas con la de las reales usando las características de un
clasificador. Pendiente, Fase 19.

**Diferencias finitas.** Comprobar un gradiente moviendo un peso un poquito
arriba y abajo y midiendo cómo cambia el error. Verifica que el *backward* es la
derivada del *forward*.

**Paridad.** Comparar nuestros números con los de PyTorch usando los mismos
pesos y las mismas entradas. Verifica que el *forward* es la arquitectura que se
pretendía.

**Mutación.** Introducir un fallo a propósito para comprobar que la prueba lo
detecta. Una prueba que nunca falla no demuestra nada.

**Control.** Un punto de comparación que dice si el resultado significa algo: una
red sin entrenar, un codificador congelado, o un compresor trivial del mismo
tamaño.

---

## Redes

**Capa densa (`Linear`).** Multiplica por una matriz y suma un vector. La
operación básica.

**Convolución (`Conv2D`).** Desliza un filtro pequeño por toda la imagen. Sirve
para imágenes porque busca el mismo patrón en cualquier posición.

**Activación.** Una función no lineal entre capas: sin ella, apilar capas sería
equivalente a una sola. ReLU, GELU y SiLU son las del proyecto.

**Normalización.** Reescalar valores intermedios para que el entrenamiento sea
estable. `LayerNorm` normaliza cada ejemplo; `GroupNorm`, por grupos de canales;
`RMSNorm` es una versión más barata.

**Bloque residual.** Una rama que calcula algo y se **suma** a la entrada en vez
de reemplazarla. El «atajo» deja pasar el gradiente intacto hacia abajo.

**Recurrente (`LSTM`).** Procesa una secuencia paso a paso arrastrando un estado.
`BiLSTM` la recorre en los dos sentidos.

---

## Transformer

**Atención.** Cada posición mira a las demás y decide a cuáles hacer caso. Es lo
que reemplazó a las recurrentes en el procesamiento de lenguaje.

**Multi-cabeza.** Varias atenciones en paralelo, cada una atenta a algo distinto.

**Causal.** Cada posición solo puede mirar hacia atrás, nunca al futuro. Es lo
que permite generar texto palabra a palabra.

**Embedding.** Tabla que convierte un token (un número) en un vector.

**Token.** La unidad mínima de texto. Aquí es **un carácter**; en modelos grandes
suele ser un fragmento de palabra (BPE).

**Contexto (`block_size`).** Cuántos tokens puede mirar el modelo a la vez.

**Posición aprendida vs RoPE.** El Transformer necesita saber el orden de los
tokens. O se aprende una tabla de posiciones, o se **rotan** los vectores según
la posición (RoPE), que hace que la atención dependa solo de la *distancia*
entre tokens.

**KV-Cache.** Al generar, guardar las claves y valores ya calculados para no
recalcular todo el texto anterior en cada token nuevo.

**Pesos compartidos (*weight tying*).** Usar la misma matriz para convertir
tokens en vectores al entrar y para producir la predicción al salir.

**Temperatura.** Al generar, cuánto azar se permite. Baja: repetitivo y seguro.
Alta: variado y con más errores.

---

## Difusión

**Difusión.** Entrenar una red para **quitar ruido**, y luego generar partiendo
de ruido puro y quitándoselo poco a poco.

**Paso de ruido (`t`).** Cuánto ruido lleva una imagen: `t = 0` es la imagen
limpia y `t = T` es ruido puro.

**Calendario (*schedule*).** La receta de cuánto ruido se añade en cada paso.
`ᾱ` (alfa barra) dice cuánta imagen original queda.

**DDPM.** El muestreador original: recorre todos los pasos, con algo de azar en
cada uno.

**DDIM.** Un muestreador que puede **saltarse pasos** y, con `eta = 0`, es
determinista: la misma semilla da siempre la misma imagen.

**U-Net.** La red de la difusión: baja de resolución para ver la forma global,
vuelve a subir, y usa **saltos** para no perder el detalle fino.

**EMA (media móvil de pesos).** Guardar un promedio suavizado de los pesos
durante el entrenamiento. En difusión se muestrea con él, porque los pesos
finales oscilan alrededor del bueno.

---

## Difusión latente

**Latente.** La versión comprimida de una imagen: aquí, 8×8×4 números en vez de
32×32.

**Autoencoder.** Un par de redes: el **codificador** comprime y el
**decodificador** reconstruye.

**Reparametrización.** Truco para poder entrenar cuando hay azar de por medio:
en vez de sortear el latente directamente, se escribe como `media + desviación ×
ruido`, y así el gradiente puede atravesarlo.

**KL (divergencia de Kullback-Leibler).** Medida de cuánto se parece una
distribución a otra. Aquí penaliza que el latente se aleje de una gaussiana
estándar, lo que mantiene el espacio ordenado.

**Escala del latente.** Un factor que se aplica antes de difundir, porque la
difusión supone datos centrados y de varianza cercana a uno.

---

## Ingeniería

**Checkpoint.** Una foto del entrenamiento: pesos, estado del optimizador y
configuración, con lo necesario para continuar exactamente donde se dejó.

**Sello.** Los metadatos que guarda un checkpoint con toda la configuración del
run. Al reanudar, manda el sello.

**Transaccional.** Escribir todos los archivos de un checkpoint como temporales
y moverlos solo cuando todos están completos, para que un corte no deje una
mezcla.

**NSF.** El formato de archivo del proyecto: número mágico, versión, metadatos,
tensores con nombre y forma, y suma de comprobación.

**Determinista / idéntico bit a bit.** Que dos ejecuciones den exactamente los
mismos números, hasta el último decimal. Es lo que permite comparar y verificar.

**Paso de un optimizador.** `AdamW` mantiene dos medias móviles por peso —del
gradiente y de su cuadrado— y las usa para adaptar el tamaño del paso.
