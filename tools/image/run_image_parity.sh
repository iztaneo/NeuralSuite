#!/usr/bin/env bash
# Copyright 2026 NeuralSuite Authors.
# Licensed under the Apache License, Version 2.0.
#
# Compara el decodificador de imagen contra Pillow. Requiere el entorno virtual
# de LLMRasec, el mismo que usan las pruebas de paridad con PyTorch.
#
# Uso:  tools/image/run_image_parity.sh [ruta-a-LLMRasec]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
LLMRASEC="${1:-$(cd "${REPO}/../LLMRasec" 2>/dev/null && pwd || echo "")}"
WORK="${TMPDIR:-/tmp}/nsimg"

if [[ -z "${LLMRASEC}" || ! -d "${LLMRASEC}" ]]; then
  echo "No se encontro el proyecto de referencia LLMRasec." >&2
  echo "Pasalo como argumento: tools/image/run_image_parity.sh /ruta/a/LLMRasec" >&2
  exit 1
fi

PY="${LLMRASEC}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Falta el entorno virtual en ${LLMRASEC}/venv." >&2
  exit 1
fi

mkdir -p "${WORK}"
# Se enlaza la biblioteca, no solo la cabecera: desde que las implementaciones
# viven en src/, compilar imgdump.cpp suelto deja los simbolos sin definir.
make -C "${REPO}" libneuralsuite.a >/dev/null
g++ -std=c++17 -O2 -pthread -I"${REPO}/include" "${HERE}/imgdump.cpp" \
    "${REPO}/libneuralsuite.a" -o "${WORK}/imgdump"

"${PY}" "${HERE}/generar_casos.py" --out "${WORK}"

# Las imagenes que ya vivian en el repositorio entran tambien: son PNG reales,
# escritos por herramientas ajenas, y no se parecen a los casos fabricados.
for f in "${REPO}"/*.png; do
  [[ -e "${f}" ]] || continue
  cp "${f}" "${WORK}/repo_$(basename "${f}")"
  "${PY}" - "${WORK}/repo_$(basename "${f}")" <<'PY'
import sys
import numpy as np
from PIL import Image
path = sys.argv[1]
img = Image.open(path)
if img.mode == "P":
    img = img.convert("RGBA" if "transparency" in img.info else "RGB")
elif img.mode == "1":
    img = img.convert("L")
a = np.array(img)
if a.ndim == 2:
    a = a[:, :, None]
np.save(path.rsplit(".", 1)[0] + ".npy", a)

# La misma imagen como JPEG. Un documento escaneado no se parece a los patrones
# sinteticos: tiene bordes duros y zonas planas, que es donde la cuantizacion
# produce los coeficientes mas dispares.
base = path.rsplit(".", 1)[0].replace("repo_", "repofoto_")
rgb = img.convert("RGB")
for etiqueta, kw in (("_420", dict(quality=85, subsampling=2)),
                     ("_prog", dict(quality=85, subsampling=2, progressive=True))):
    destino = base + etiqueta + ".jpg"
    rgb.save(destino, "JPEG", **kw)
    np.save(destino[:-4] + ".npy", np.array(Image.open(destino)))
PY
done

"${PY}" "${HERE}/compare_pillow.py" --dir "${WORK}" --tool "${WORK}/imgdump"
