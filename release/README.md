# release/

Directorio de artefactos generados por entrenamiento: pesos de modelos
(`*.ns`, `*.bin`) y vocabularios (`vocab_*.txt`).

## Regla

**Todo modelo entrenado se escribe aquí, nunca en la raíz del repositorio.**

El contenido de este directorio está excluido del control de versiones (ver
`.gitignore`); solo se versionan este README y el `.gitkeep`. Los checkpoints
que se quieran distribuir se adjuntan a un [GitHub Release][releases], que es
el mecanismo pensado para binarios: no infla el historial de git ni el tamaño
de cada clon.

[releases]: https://github.com/iztaneo/NeuralSuite/releases

## Cómo escriben aquí los ejecutables

La ruta la resuelve `ReleasePath()` de [`include/artifacts.h`](../include/artifacts.h),
que además crea el directorio si no existe:

```cpp
#include "artifacts.h"

const std::string path = ReleasePath("mi_modelo.ns");  // -> "release/mi_modelo.ns"
if (!model.Save(path)) {
  std::cerr << "ERROR: no se pudo guardar en '" << path << "'.\n";
}
```

Las rutas son relativas al directorio de trabajo, así que los ejecutables deben
lanzarse desde la raíz del repositorio para que `release/` sea el mismo
directorio en todos los casos.

## Artefactos que producen los ejecutables

| Ejecutable          | Artefacto                                      |
| ------------------- | ---------------------------------------------- |
| `train_llm`         | `model_cpp.bin`, `vocab_cpp.txt`               |
| `demo_autoencoder`  | `encoder_model.ns`, `decoder_model.ns`         |
| `demo_gan`          | `generator_model.ns`, `discriminator_model.ns` |
| `demo_diffusion`    | `diffusion_denoiser.ns`                        |
| `demo_resnet`       | `resnet_model.ns`                              |
| `demo_sequential`   | `sequential_model.ns`                          |
| `demo_ocr`          | `ocr_model.ns`                                 |

`generate_llm` y `ocr_cli` **leen** de aquí en lugar de escribir.

## Nota sobre el formato

Los archivos `.ns` / `.bin` actuales son un volcado crudo de floats, sin
cabecera, versión, arquitectura ni checksum: cargar un checkpoint que no
corresponda a la arquitectura configurada no produce ningún error, solo datos
truncados o basura. Reemplazarlo por un formato versionado es trabajo
pendiente de una fase posterior.
