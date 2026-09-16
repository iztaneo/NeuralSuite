# Guía: el autoencoder de la difusión latente (LDM-1)

La difusión de la [guía anterior](GUIA_DIFUSION.md) trabaja directamente sobre
los píxeles, y eso es caro. La idea del artículo de *Latent Diffusion* (Rombach
et al.) es hacerlo en dos tiempos:

```
imagen 32×32  ──comprimir──►  latente 8×8  ──difundir ahí──►  descomprimir  ──►  imagen
```

Este documento cubre la primera parte: **el compresor**. Es un autoencoder
convolucional con latente gaussiano, entrenado sobre MNIST rellenado a 32×32.

**Por qué merece la pena, medido** en esta máquina, con la misma U-Net:

| Entrada de la U-Net | ms por paso |
| --- | --- |
| Latente 8×8 | **40** |
| Píxeles 32×32 | 252 |

Difundir en el latente es **6.3 veces más barato**.

---

## 1. Entrenar

```bash
caffeinate -i ./bin/train_autoencoder --n_imagenes 0 --n_validacion 3000 --c_lat 4 --lote 32 --iteraciones 18000 --calentamiento 500 --lr 5e-4 --lr_min 1e-5 --evaluar_cada 1000 --n_eval 1000 --guardar_cada 1000 --archivar_cada 3000 --reportar_cada 250 --archivo release/ae_c4.nsf --png release/ae_c4.png > logs/ae_c4.log 2>&1
```

Unos 78 minutos (10 épocas sobre 57 000 imágenes). Comparte toda la
infraestructura del entrenador de difusión: checkpoints sellados, `--reanudar`
sin repetir opciones, `--parar_en`, copias archivadas y partición reproducible.

| Opción | Por defecto | Qué hace |
| --- | --- | --- |
| `--c_lat` | 4 | Canales del latente. El latente tiene `8 × 8 × C` números |
| `--canales` | 32 | Anchura de la red |
| `--peso_kl` | 1e-6 | Peso de la penalización KL, el del artículo |
| `--n_eval` | 1000 | Imágenes de validación por evaluación |
| `--evaluar_cada` | — | Evalúa y actualiza el PNG de reconstrucciones |
| `--medir_escala` | — | No entrena: mide la escala del latente y la sella (ver abajo) |

El resto (`--n_imagenes`, `--n_validacion`, `--particion`, `--lote`, `--lr`,
`--lr_min`, `--calentamiento`, `--guardar_cada`, `--archivar_cada`,
`--parar_en`, `--reanudar`, `--semilla`) funciona igual que en `train_diffusion`.

### La puerta de sobreajuste, primero

Antes del run largo conviene comprobar que el compresor puede memorizar:

```bash
./bin/train_autoencoder --n_imagenes 16 --iteraciones 600 --lote 16 --lr_min 1e-3
```

En unos 90 segundos debe pasar de ~2 dB a **más de 31 dB**. Si no lo consigue,
hay un defecto y el entrenamiento largo no lo arreglará.

---

## 2. Cómo se juzga: contra un compresor tonto

El PSNR por sí solo engaña. Un codificador **sin entrenar** ya alcanza 21 dB
memorizando 16 imágenes, así que un umbral fijo no distingue aprender de no
aprender.

Por eso cada evaluación se compara con un compresor trivial que guarda **los
mismos números que el latente**: reducir la imagen promediando y volver a
ampliarla. El log lo imprime como `referencia` y la diferencia como `ventaja`.

---

## 3. Resultados: el barrido de `C`

Tres entrenamientos idénticos salvo por los canales del latente, 18 000
iteraciones cada uno, evaluados sobre 1 000 imágenes de validación:

| `C` | Números del latente | Compresión | Referencia | **PSNR validación** | Ventaja |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 16× | 15.01 dB | 29.35 dB | +14.34 dB |
| 2 | 128 | 8× | 16.45 dB | 30.90 dB | +14.46 dB |
| **4** | **256** | **4×** | 19.46 dB | **33.23 dB** | +13.77 dB |

Los tres superan con claridad a su referencia, y en los tres el PSNR de
entrenamiento y el de validación van a la par (diferencia menor de 0.1 dB): el
compresor **generaliza**, no memoriza.

**La elección para la Fase 19 es `C = 4`**, y la razón es que difundir cuesta
prácticamente lo mismo con cualquier `C`: 39.3 ms con C = 1 frente a 40.3 ms con
C = 4, porque el coste de la U-Net lo domina su anchura interna y no los canales
de entrada. Con el mismo coste, conviene la mejor reconstrucción.

Esa reconstrucción es el **techo de calidad** de todo lo que venga después: la
Fase 19 no podrá generar nada mejor de lo que el decodificador sabe reconstruir.

---

## 4. Sellar la escala del latente

```bash
./bin/train_autoencoder --medir_escala --archivo release/ae_c4.nsf
```

No entrena. Codifica la validación, mide la distribución del latente y guarda el
resultado en el sello de los tres archivos del checkpoint. Con el modelo de
referencia:

```
latente sobre 1000 imagenes: media -0.6936, sigma 0.6090
sellado para la Fase 19: (z - -0.693598316) * 1.64210055
la UNet2D acepta el latente 8x8x4 y devuelve [2,4,8,8]  ok
```

**Por qué hace falta.** La difusión supone datos centrados y de varianza cercana
a uno; su calendario de ruido está calibrado para eso. El latente de un
autoencoder con KL débil no tiene por qué cumplirlo: aquí la penalización apenas
regulariza y el latente se expande durante el entrenamiento. Difundir sobre un
latente descentrado no da ningún error, solo imágenes peores.

**Una diferencia con el artículo:** allí basta con dividir por σ, porque sus
latentes ya salen centrados. Aquí la media es −0.69 con σ = 0.61, así que
reescalar sin centrar dejaría el latente desplazado más de una desviación. Por
eso se sellan **media y escala**, y la Fase 19 usará `(z − media) × escala`.

---

## 5. Qué produce

| Archivo | Contenido |
| --- | --- |
| `release/ae_c4.nsf` | Pesos del codificador, con la arquitectura y la escala en los metadatos |
| `release/ae_c4.nsf.dec` | Pesos del decodificador |
| `release/ae_c4.nsf.opt` | Estado del optimizador, para reanudar |
| `release/ae_c4.png` | Originales y reconstrucciones, en filas alternas |
| `release/ae_c4_itNNNNNN.*` | Copias archivadas |

---

## Límites actuales

- **La pérdida es cuadrática**, no la combinación de L1, pérdida perceptual y GAN
  del artículo. La perceptual necesitaría una VGG preentrenada, y esa decisión
  está aplazada: importar pesos ajenos rompería la premisa del proyecto.
- **Solo MNIST en blanco y negro a 32×32.** El factor de compresión es 4 (de
  32×32 a 8×8), como en el artículo.
- **Todavía no genera nada.** Eso es la Fase 19: difundir sobre este latente y
  comparar, con el mismo presupuesto, contra la difusión en píxeles.
