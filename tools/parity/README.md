# Pruebas de paridad contra PyTorch

Comparan NeuralSuite con la implementación de referencia en PyTorch del
proyecto [LLMRasec](https://github.com/iztaneo/LLMRasec): mismos pesos, misma
entrada, y se contrastan salida y gradientes.

## Por qué existen, además del gradient checking

Los gradient checks de `test_suite` verifican que el `Backward` de cada capa
derive correctamente **el `Forward` que se escribió**. Comparan el código
consigo mismo, así que no pueden detectar que el `Forward` implemente algo
distinto de lo que dice implementar.

Eso no es hipotético. La capa `LSTM` de este repositorio calculaba

    h = tanh(x[0] + h)

sin usar ninguno de sus cuatro tensores de parámetros, sin puertas y sin estado
de celda. Un gradient check escrito para ese `Forward` habría pasado
perfectamente. Comparada contra `nn.LSTM`, esa misma versión falla las siete
comprobaciones, con una pérdida de −7.53 donde PyTorch da 0.46.

PyTorch actúa aquí como oráculo externo: sabe cómo *debe* ser un LSTM. Es la
única capa de verificación que detecta errores de semántica.

## Requisitos

El proyecto de referencia con su entorno virtual:

```bash
cd ../LLMRasec
python3 -m venv venv
./venv/bin/pip install torch numpy pillow tqdm
```

## Ejecución

```bash
tools/parity/run_parity.sh [ruta-a-LLMRasec]
```

Compila los binarios de paridad, exporta las referencias desde PyTorch, ejecuta
el lado C++ y compara. Devuelve un código de salida distinto de cero si alguna
comparación queda fuera de tolerancia.

## Qué se compara

| Caso   | Cubre                                                                 |
| ------ | --------------------------------------------------------------------- |
| `gpt`  | Pérdida, logits y los 28 gradientes de parámetros del `GPTModel`       |
| `lstm` | Pérdida, secuencia de salida, `dx` y los 4 gradientes frente a `nn.LSTM` |
| `bilstm` | Lo mismo con `bidirectional=True`: 8 gradientes, y cada mitad de la salida por separado |
| `crnn` | El OCR completo: logits, `dx` hacia la imagen y los 16 gradientes frente a `src/ocr.py` |

## Diferencias que se controlan

Para que la comparación sea justa hay que neutralizar todo lo que no sea el
cálculo bajo prueba:

- **Dropout.** El modelo de referencia usa 0.1 por defecto. Se fuerza a 0 y se
  llama a `model.eval()`; en otro caso el forward es estocástico.
- **GELU.** C++ usa la aproximación por tanh y `nn.GELU()` usa por defecto la
  formulación exacta con erf. El exportador sustituye por
  `nn.GELU(approximate="tanh")` para aislar esa diferencia.
- **Disposición de los pesos densos.** `nn.Linear` guarda `[out, in]` y calcula
  `x·Wᵀ`; la capa `Linear` de C++ guarda `[in, out]` y calcula `x·W`. El
  exportador transpone todas las matrices densas.
- **LSTM.** Aquí no hace falta convertir nada: el orden de puertas (`i, f, g,
  o`), la disposición de los pesos y el layout `[seq, batch, input]` ya
  coinciden con los de `nn.LSTM`.
- **BiLSTM.** Tampoco. Los parámetros del sentido inverso son los que PyTorch
  llama `*_l0_reverse`, y la salida concatena directa e inversa en ese orden en
  ambas implementaciones. Las dos mitades se comparan además por separado: si
  se hubieran intercambiado al concatenar, el error agregado podría quedar
  disimulado por el de la otra mitad.
- **Conv2D.** `nn.Conv2d` guarda `[out, in, kh, kw]`, igual que `Conv2D`, así
  que aquí tampoco hay nada que convertir. La capa densa final del CRNN sí se
  transpone, como todas las densas.

## El caso del CRNN

Es el único donde la tolerancia de `1e-3` no basta, y conviene saber por qué. El
gradiente de un peso de convolución es una suma sobre todas las posiciones del
lote y de la imagen —miles de términos—, y el orden en que se acumulan no
coincide entre las dos implementaciones. En float32 eso se nota: `conv2.weight`
sale en 2.3e-03.

Por eso `export_crnn.py` calcula además el mismo modelo en float64 y guarda esos
gradientes como referencia. `compare_crnn.py` solo declara un fallo si C++ se
aparta de ese valor mucho más que PyTorch en float32. En la configuración
actual, ambos se apartan por igual —6.1e-03 y 8.3e-03—, así que la discrepancia
es redondeo. Es el mismo criterio que `precision_probe.py` aplica al GPT, y la
razón por la que ninguno de los dos usa un umbral fijo a ojo.

## Sobre la tolerancia

El umbral por defecto es `1e-3`, y no es arbitrario. `precision_probe.py`
recalcula el mismo modelo en float64 y mide a qué distancia queda cada
implementación float32 de ese valor: en esta configuración, PyTorch se desvía
hasta 1.79e-04 y C++ hasta 1.67e-04. Es decir, ambas acumulan redondeo en la
misma medida, y C++ queda incluso marginalmente más cerca.

Un umbral más estricto marcaría ese redondeo como defecto. Los defectos reales
de gradiente que hemos encontrado aparecen con error relativo entre 1e-1 y 1.0,
tres órdenes de magnitud por encima del límite.

Cuando una comparación falle, `precision_probe.py` es la herramienta para
decidir si es un defecto o precisión.

## Formato de intercambio

`nsparity.py` define un contenedor mínimo (`NSPARITY`) con cabecera, versión y
tensores con nombre y forma. Existe solo para que ambas implementaciones carguen
exactamente los mismos datos; **no** es el formato de checkpoints del proyecto,
que sigue pendiente de diseñar.
