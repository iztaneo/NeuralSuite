# Guía: entrenar un modelo de difusión y generar imágenes

NeuralSuite entrena un modelo de difusión (DDPM) con una U-Net y genera dígitos
manuscritos que no existen en el conjunto de datos. Es el mismo principio que
usan los generadores de imágenes modernos, a la escala que permite una CPU.

Todos los comandos se lanzan desde la raíz del repositorio, después de `make`.

---

## 1. Descargar MNIST

```bash
python3 tools/data/descargar_mnist.py
```

Descarga unos 11 MB y deja los cuatro archivos IDX en `corpus/mnist/`,
comprobando tamaño y suma MD5. Es el único paso que usa Python.

---

## 2. La puerta de sobreajuste (minutos)

Antes de gastar horas conviene comprobar que el modelo es capaz de **memorizar**
unas pocas imágenes. Si no puede, hay un defecto, y ningún entrenamiento largo lo
arreglará:

```bash
./bin/train_diffusion --n_imagenes 16 --iteraciones 5000 --lote 8 --lr_min 2e-3
```

`--lr_min 2e-3` mantiene la tasa constante: con el valor por defecto (0) el
coseno la lleva a cero, y en un run corto eso frena el final. Con 16 imágenes y
200 pasos de ruido, en unos minutos
tiene que generar, desde ruido puro, dígitos casi idénticos a esos 16. El
programa lo mide con la distancia a la imagen más cercana del conjunto.

---

## 3. El entrenamiento completo (unas 5 horas)

El modelo de referencia (`release/unet_mnist.nsf`) se entrenó así:

```bash
caffeinate -i ./bin/train_diffusion --n_imagenes 0 --n_validacion 3000 --canales 32 --pasos 1000 --lote 32 --iteraciones 80000 --calentamiento 1000 --lr 2e-4 --lr_min 1e-5 --ema 0.9995 --evaluar_cada 2000 --muestrear_cada 5000 --guardar_cada 2000 --archivar_cada 10000 --reportar_cada 250 --archivo release/unet_mnist.nsf > logs/difusion_mnist.log 2>&1
```

`caffeinate -i` es para macOS: impide que el equipo se duerma y congele el run.
En Linux se puede omitir. Para seguir el progreso:

```bash
tail -f logs/difusion_mnist.log
```

| Opción | Por defecto | Qué hace |
| --- | --- | --- |
| `--n_imagenes` | 16 | Imágenes a usar; `0` = todas |
| `--n_validacion` | 0 | Imágenes apartadas para validación |
| `--particion` | `barajada` | `barajada` (con `DataLoader`) o `contigua` (las últimas) |
| `--canales` | 32 | Anchura de la U-Net |
| `--pasos` | 200 | Pasos de ruido del calendario |
| `--beta_fin` | automático | Si no se da, se escala por `1000/pasos` |
| `--lote`, `--iteraciones` | 8, 400 | |
| `--lr`, `--lr_min`, `--calentamiento` | 2e-3, 0, 0 | Calentamiento lineal y después coseno hasta `lr_min` |
| `--ema` | 0.999 | Decaimiento de la media de los pesos (EMA) |
| `--evaluar_cada` | — | Pérdida de entrenamiento y validación |
| `--muestrear_cada` | — | Dibuja un dígito generado en el log |
| `--guardar_cada` | — | Checkpoint que se sobrescribe |
| `--archivar_cada` | — | Copia aparte que no se sobrescribe |
| `--parar_en` | — | Corta antes sin alterar el plan (ver abajo) |
| `--reanudar` | — | Continúa desde `--archivo` |

**Sobre `--beta_fin`.** Los valores del artículo están pensados para 1000 pasos.
Con menos pasos y los mismos valores, la imagen no llega a convertirse en ruido
puro y el modelo genera manchas sin dar ningún error. Por eso el valor se ajusta
solo, y el log imprime cuánta imagen queda al final del proceso de ruido.

### Qué produce

Tres archivos por checkpoint: los pesos (`.nsf`), la media de los pesos (`.ema`)
y el estado del optimizador (`.opt`). Las copias archivadas llevan el número de
iteración, como `unet_mnist_it010000.nsf`.

---

## 4. Cortar y reanudar

Un run largo se puede interrumpir y continuar **con un resultado idéntico bit a
bit** al de no haber parado:

```bash
./bin/train_diffusion --reanudar --archivo release/unet_mnist.nsf
```

No hace falta repetir ninguna opción. **Todo lo que decide la trayectoria** (datos,
partición, semilla, tamaño, tasa, calendario) **va sellado en el checkpoint** y se
adopta al reanudar. Si se pasa una opción que contradice el sello, el programa
aborta explicando la diferencia.

Tres detalles importantes:

- **`--parar_en`, no `--iteraciones`, para cortar.** El coseno de la tasa se
  calcula sobre `--iteraciones`: bajarlo daría otro entrenamiento. `--parar_en`
  corta sin tocar el plan.
- **El total no se puede ampliar después**, por la misma razón.
- **Los tres archivos se escriben de forma transaccional** y con un identificador
  común. Si un corte los mezcla, reanudar se niega en vez de continuar mal.

Se puede reanudar desde cualquier copia archivada copiando sus tres archivos.

---

## 5. Generar imágenes

```bash
./bin/sample_diffusion --muestreador ddpm --n 64 --salida release/muestras_ddpm.png
```

```bash
./bin/sample_diffusion --muestreador ddim --pasos_ddim 100 --n 64 --salida release/muestras_ddim.png
```

Guarda una rejilla PNG y dice, para cada muestra, a qué dígito del conjunto se
parece más y cuánto. Por defecto usa la media de los pesos (`release/unet_mnist.nsf.ema`).

| Opción | Por defecto | Qué hace |
| --- | --- | --- |
| `--muestreador` | `ddim` | `ddpm`: recorre los 1000 pasos; `ddim`: una subsecuencia, mucho más rápido |
| `--pasos_ddim` | 100 | Pasos de DDIM (al menos 2) |
| `--eta` | 0 | DDIM: 0 determinista; 1 equivale a DDPM |
| `--n` | 64 | Número de muestras |
| `--semilla` | 2026 | Misma semilla, mismas imágenes |
| `--escala` | 4 | Aumento de cada dígito en el PNG |
| `--sin_recorte` | — | Diagnóstico: no recortar el `x₀` estimado |

El calendario de ruido se lee del sello del checkpoint. Con un checkpoint anterior
al sello, como el de referencia, el programa **avisa** de que asume 1000 pasos, que
es lo que se usó.

### Resultados con el modelo de referencia

Sobre 64 muestras:

| | DDPM, 1000 pasos | DDIM, 100 pasos |
| --- | --- | --- |
| Tiempo | 113.6 s | 10.6 s |
| Clases cubiertas | 10 de 10 | 10 de 10 |
| Distancia a la vecina más cercana / distancia media | 0.50 | 0.56 |

Con DDPM, unos 45 de los 64 dígitos se leen sin dudar. DDIM tuvo un fallo en
el recorte que lo hacía empeorar al añadir pasos; está corregido, y ahora da el
mismo resultado con 100 que con 500 pasos. El cociente de distancias
está lejos del 0.15–0.35 que daba el modelo cuando memorizaba 16 imágenes: **genera
dígitos nuevos, no copia**.

---

## Qué esperar, y qué no

- Los dígitos son **manuscritos plausibles, no perfectos**: una parte sale ambigua.
- **Todo es en blanco y negro y a 28×28.** Es el tamaño que se puede entrenar en
  CPU en horas.
- **No se puede pedir un dígito concreto.** Generar el que se pida (condicionar)
  y trabajar en un espacio comprimido (difusión latente) son los siguientes
  pasos del proyecto.
