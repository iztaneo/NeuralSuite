# Evaluación del OCR

```bash
../LLMRasec/venv/bin/python tools/ocr/evaluar.py --pesos release/ocr_texto.ns
```

Mide el error de carácter sobre imágenes reales que el modelo nunca vio, y lo
pone al lado de lo que saca Tesseract sobre las mismas. Devuelve 1 si algo
impide que la medición valga.

## Por qué comprueba antes de medir

Existe por un fallo concreto, y conviene que quede escrito. Las mediciones se
hacían a mano con un fragmento de Python, y una de ellas ejecutó `ocr_cli` con
`2>/dev/null`. Eso tiró este aviso:

```
Error al cargar: num_classes del archivo es 92 y el modelo esperaba 91.
AVISO: el modelo corre con inicializacion aleatoria y lo que salga
       no significara nada.
```

Se midió la salida de una red sin entrenar y dio 98% de error. La conclusión que
casi se reporta era que añadir acentos al vocabulario había empeorado el modelo
doce veces. La causa real era que `ocr_cli` construía el modelo con una clase de
menos.

El sistema hizo lo correcto: el formato NSF detectó el desajuste y se negó a
cargar. Lo que falló fue la medición, por tirar a la basura justo la parte que
decía si la medición valía.

Por eso esta herramienta comprueba, **antes de dar ninguna cifra**, el código de
salida de `ocr_cli` y la presencia de avisos de carga en su salida de error. Un
error de pesos aborta con un mensaje en vez de producir un número.

## Los casos

| | Qué prueba |
| --- | --- |
| `iliada_libro1.png` | Página de libro: serif pequeña, 18 renglones, acentos y puntuación |
| `test_image.png` | Logotipo con dos palabras debajo |

Las referencias están en `docs/referencias/`. La de la Ilíada se escribió
leyendo la imagen y **se verificó contra Tesseract**: 16 de sus 18 líneas
coinciden carácter a carácter, y las dos discrepancias son `LIBRO I` frente a
`LIBRO 1` y `Leto;` frente a `Leto:` — dos caracteres de 853.

## `solo_texto`

El caso del logotipo lleva esa marca. El cortador de renglones entrega también
las dos bandas del dibujo, y el reconocedor devuelve basura sobre ellas porque
nunca vio ejemplos negativos: distinguir texto de dibujo sigue pendiente en el
roadmap.

Medir la página entera mezclaría dos fallos distintos —lo que el reconocedor no
sabe leer y lo que el detector no sabe descartar— en una sola cifra. Con
`solo_texto`, cada renglón de la referencia se compara con el que mejor le
encaja, lo que aísla el reconocimiento. Da 0.0%: `MITSUBISHI` y `MOTORS` salen
perfectos.

Cuando existan los ejemplos negativos, esa marca podrá quitarse y la cifra
medirá el canal completo.
