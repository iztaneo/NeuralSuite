# Cómo se verifica NeuralSuite

Un framework de redes neuronales puede estar mal sin que nada falle. La pérdida
baja igual con un gradiente que llega a medias, una arquitectura mal cableada
entrena, y un muestreador incoherente produce imágenes con aspecto razonable.
Por eso NeuralSuite no se da por bueno cuando compila ni cuando entrena, sino
cuando pasa **varias capas de verificación independientes**. Cada una ve
errores que las demás no pueden ver.

Este documento explica cada capa, cómo ejecutarla y **qué encontró de verdad**.
El historial completo, fase por fase, está en [ROADMAP.md](ROADMAP.md).

---

## 1. Diferencias finitas: el backward deriva el forward

Cada capa implementa su derivada (`Backward`). Para comprobarla se perturba cada
entrada un poco hacia arriba y hacia abajo, se mide cuánto cambia la pérdida, y
se compara ese gradiente numérico con el analítico.

**Qué garantiza:** que `Backward` es la derivada exacta de `Forward`.

**Qué no puede ver:** si `Forward` calcula lo que debería. Una red cableada al
revés, pero de forma coherente, deriva perfectamente su propio forward
equivocado.

Está en la suite (`./bin/test_suite`, 60 pruebas), que cubre desde `Linear` y
`Conv2D` hasta la `UNet2D` completa, el latente gaussiano y la infraestructura de
checkpoint.

---

## 2. Paridad contra PyTorch: el forward es lo que dice ser

Los mismos pesos y las mismas entradas se pasan por NeuralSuite y por una
implementación de referencia en PyTorch
([LLMRasec](https://github.com/iztaneo/LLMRasec)), y se comparan salidas y
gradientes con tolerancia de `1e-3` relativo. En la práctica los errores quedan
muy por debajo: casi todos alrededor de `1e-6`, y el peor caso actual es
`1.7e-4`, en DDPM con recorte, por el redondeo de `float32` acumulado en veinte
pasos.

**Qué garantiza:** que la arquitectura es la que se pretendía, y no solo una
coherente.

Casos actuales:

| Caso | Qué compara |
| --- | --- |
| `gpt` | El GPT completo: pérdida, logits y gradientes |
| `lstm`, `bilstm` | Las recurrentes, paso a paso |
| `crnn` | El modelo de OCR |
| `bloques` | RMSNorm, SiLU, GroupNorm, remuestreo 2D, CrossAttention, SwiGLU, RoPE, calendario de difusión, embedding del tiempo, bloque residual con tiempo, `UNet2D`, DDPM y DDIM (con y sin recorte), `ResBlock2D`, codificador y decodificador del autoencoder, y el latente gaussiano con su KL |

Además, los decodificadores de imagen (PNG, JPEG, BMP, Netpbm) se comparan
contra Pillow.

```bash
tools/parity/run_parity.sh /ruta/a/LLMRasec
```

```bash
tools/image/run_image_parity.sh /ruta/a/LLMRasec
```

Requiere el entorno virtual de LLMRasec con `torch`, `numpy` y `pillow`. Ver
[tools/parity/README.md](../tools/parity/README.md).

**Ejemplo de por qué hacen falta las dos capas.** Se movió de sitio una de las
subidas del decodificador del autoencoder, en el forward y en el backward a la
vez. La prueba de diferencias finitas **siguió en verde**: el cableado era
coherente. La paridad lo rechazó con un error relativo de **2.2**.

---

## 3. Mutaciones: la prueba detecta el fallo

Una prueba que nunca falla no demuestra nada. Así que, al escribir cada prueba
importante, se introducen **fallos deliberados** en el código y se exige que la
prueba se ponga en rojo. Si no lo hace, la prueba no sirve y se reescribe.

**Lo que encontró.** Para los muestreadores de difusión se escribió primero una
prueba que parecía muy fuerte: con un predictor perfecto, el resultado debe
coincidir exactamente con la imagen esperada. **Detectaba 1 de 4 mutaciones.** El
predictor perfecto se corrige solo en cada paso, así que fija el punto de
llegada pero no el camino. Hubo que añadir pruebas sobre la trayectoria.

Otros ejemplos que se probaron con mutaciones: olvidar el ½ al derivar
`exp(logvar/2)`, no dividir la KL por el tamaño del lote, quitar el gradiente
del atajo de un bloque residual, o sobrescribir en vez de sumar los dos caminos
que llegan a un salto de la U-Net.

---

## 4. Casos con respuesta exacta

Siempre que existe, se usa un caso cuya respuesta se conoce sin calcularla con el
propio código:

- Con media 0 y varianza 1, la KL del latente vale **exactamente** 0.
- Con un predictor que devuelve ruido cero, el producto de escalas de DDIM
  tiene que dar `1/√ᾱ` para **cualquier** número de pasos.
- DDIM con `eta = 1` sobre la secuencia completa tiene que reproducir DDPM paso a
  paso.

---

## 5. Reproducibilidad exacta

- **Paralelismo sin cambiar un bit.** Las operaciones que reparten trabajo entre
  hilos dan el mismo resultado con 1 hilo que con 10, porque cada hilo escribe
  zonas que ningún otro toca.
- **Reanudar es idéntico a no parar.** Cada iteración se siembra solo con
  `(semilla, iteración)`, y el checkpoint guarda pesos, EMA y el estado de Adam.
  Probado con el modelo real de difusión: reanudar desde la iteración 70 000
  hasta la 80 000 dio pesos, EMA y momentos de Adam **idénticos bit a bit** a los
  del run sin interrupción.
- **Refactorizar sin cambiar resultados.** Cuando se extrajo la infraestructura de
  checkpoint a la biblioteca, se exigió que el entrenador produjera exactamente
  los mismos pesos que el binario anterior.

---

## 6. Controles en los experimentos

Un número bueno no significa nada si un modelo que no aprendió daría el mismo
número. Cada experimento lleva su control:

| Experimento | Control | Qué reveló |
| --- | --- | --- |
| Difusión, reconstrucción desde `t = 20` | Una red **sin entrenar** | También dibujaba el dígito: a `t = 20` la imagen ya está casi limpia. La prueba se movió a `t = 140`. |
| Autoencoder, puerta de 16 imágenes | Codificador **congelado** | Se atascaba en 21 dB frente a 32 dB. Probó que el codificador aprende, y que un umbral fijo de PSNR engañaría. |
| Autoencoder sobre MNIST | Reducir y ampliar con **los mismos números que el latente** | El listón está en 15–19 dB. C = 4 llega a 33.2 dB. |
| Difusión generada | Distancia a la imagen más cercana del conjunto | Las muestras están lejos de cualquier imagen de entrenamiento: generan, no copian. |

**Un resultado imposible se trata primero como fallo del medidor.** Una
evaluación dio las 64 muestras en la misma clase con distancia 0.0000; no era el
modelo, era un `ParallelFor` mal llamado que no calculaba nada.

---

## 7. Integración continua

Cada commit en `main` ejecuta:

- Compilación y pruebas en **Linux** (GCC y Clang), **macOS** (AppleClang) y
  **Windows** (MSVC), en `Debug` y `Release`.
- **Paridad contra PyTorch** y contra Pillow.
- La suite bajo **AddressSanitizer** y **UndefinedBehaviorSanitizer**.
- Las demos y un entrenamiento del LLM de extremo a extremo.

**Lo que encontró.** Dos fallos que solo se manifiestan en Windows. El último:
`std::rename` **falla si el destino ya existe**, mientras que en POSIX lo
reemplaza, así que el primer checkpoint de un entrenamiento se escribía y todos
los siguientes fallaban. Se encontró al cerrar la Fase 18, con la prueba del
guardado transaccional. Antes, nueve pruebas escribían en `/tmp`, que no existe
en Windows. El CI estuvo en rojo diecisiete horas sin que se viera, porque cada push
cancelaba el run anterior antes de que terminara el trabajo de Windows. Ahora las
pruebas usan una ruta temporal portable, y en `main` los runs ya no se cancelan.

---

## Ejecutarlo en local

```bash
make test
```

```bash
tools/parity/run_parity.sh /ruta/a/LLMRasec
```

Los sanitizers conviene dejárselos al CI si trabajas en macOS 26: su versión de
AddressSanitizer se queda colgada al arrancar, incluso con un «hola mundo».
