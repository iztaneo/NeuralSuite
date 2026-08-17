"""Formato de intercambio para las pruebas de paridad entre PyTorch y C++.

Un contenedor mínimo y autodescriptivo: cabecera con número mágico y versión,
seguida de tensores con nombre, forma y datos en float32 little-endian. Existe
para que ambas implementaciones puedan cargar exactamente los mismos pesos y las
mismas entradas; no pretende ser el formato de checkpoints del proyecto.

    magic    8 bytes  "NSPARITY"
    version  int32
    count    int32
    por tensor:
        name_len  int32
        name      name_len bytes (ASCII)
        ndim      int32
        dims      int32 * ndim
        data      float32 * prod(dims)
"""

import struct

MAGIC = b"NSPARITY"
VERSION = 1


def write(path, tensors):
    """Escribe un dict {nombre: array numpy} en `path`."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", VERSION, len(tensors)))
        for name, array in tensors.items():
            data = array.astype("<f4", copy=False).reshape(-1)
            encoded = name.encode("ascii")
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<i", array.ndim))
            for d in array.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(data.tobytes())


def read(path):
    """Lee un archivo escrito por `write` y devuelve {nombre: array numpy}."""
    import numpy as np

    out = {}
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: no es un archivo NSPARITY")
        version, count = struct.unpack("<ii", f.read(8))
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, se esperaba {VERSION}")
        for _ in range(count):
            (name_len,) = struct.unpack("<i", f.read(4))
            name = f.read(name_len).decode("ascii")
            (ndim,) = struct.unpack("<i", f.read(4))
            dims = struct.unpack("<" + "i" * ndim, f.read(4 * ndim))
            n = 1
            for d in dims:
                n *= d
            data = np.frombuffer(f.read(4 * n), dtype="<f4").reshape(dims)
            out[name] = data.copy()
    return out
