#!/usr/bin/env bash
# Copyright 2026 NeuralSuite Authors.
# Licensed under the Apache License, Version 2.0.
#
# Ejecuta todas las pruebas de paridad contra la implementación de referencia en
# PyTorch. Requiere el proyecto LLMRasec con su entorno virtual ya instalado:
#
#     cd <LLMRasec> && python3 -m venv venv && ./venv/bin/pip install torch numpy pillow tqdm
#
# Uso:  tools/parity/run_parity.sh [ruta-a-LLMRasec]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
LLMRASEC="${1:-$(cd "${REPO}/../LLMRasec" 2>/dev/null && pwd || echo "")}"
WORK="${TMPDIR:-/tmp}/nsparity"

if [[ -z "${LLMRASEC}" || ! -d "${LLMRASEC}" ]]; then
  echo "No se encontro el proyecto de referencia LLMRasec." >&2
  echo "Pasalo como argumento: tools/parity/run_parity.sh /ruta/a/LLMRasec" >&2
  exit 1
fi

PY="${LLMRASEC}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Falta el entorno virtual en ${LLMRASEC}/venv." >&2
  echo "Crealo con: cd '${LLMRASEC}' && python3 -m venv venv &&" >&2
  echo "            ./venv/bin/pip install torch numpy pillow tqdm" >&2
  exit 1
fi

mkdir -p "${WORK}"
CXXFLAGS=(-std=c++17 -I"${REPO}/include" -O2 -DNDEBUG)
SRC=("${REPO}/src/tensor.cpp" "${REPO}/src/tokenizer.cpp")

status=0

run_case() {
  local name="$1"
  echo
  echo "### Paridad: ${name} ###"
  g++ "${CXXFLAGS[@]}" "${HERE}/parity_${name}.cpp" "${SRC[@]}" -o "${WORK}/parity_${name}"
  "${PY}" "${HERE}/export_${name}.py" --out "${WORK}/${name}_ref.nsp" >/dev/null
  "${WORK}/parity_${name}" --in "${WORK}/${name}_ref.nsp" --out "${WORK}/${name}_cpp.nsp" >/dev/null
  if ! "${PY}" "${HERE}/compare_${name}.py" \
        --ref "${WORK}/${name}_ref.nsp" --cpp "${WORK}/${name}_cpp.nsp"; then
    status=1
  fi
}

run_case gpt
run_case lstm

echo
echo "### Sonda de precision (GPT) ###"
"${PY}" "${HERE}/precision_probe.py" \
    --llmrasec "${LLMRASEC}" \
    --ref "${WORK}/gpt_ref.nsp" --cpp "${WORK}/gpt_cpp.nsp" || status=1

echo
if [[ ${status} -eq 0 ]]; then
  echo "PARIDAD COMPLETA: todas las comparaciones dentro de tolerancia."
else
  echo "PARIDAD: hay discrepancias, revisar la salida anterior."
fi
exit ${status}
