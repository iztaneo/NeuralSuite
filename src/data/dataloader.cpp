// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de data/dataloader.h.

#include "data/dataloader.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace neuralsuite {
namespace data {

namespace {

// Generador propio, minimo y reproducible. No se usa el global a proposito:
// compartirlo haria que el barajado y la inicializacion de pesos se estorbaran.
uint32_t Siguiente(uint32_t* estado) {
  uint32_t x = *estado;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *estado = x;
  return x;
}

}  // namespace

DataLoader::DataLoader(Tensor entradas, Tensor objetivos, int lote, bool barajar,
                       uint32_t semilla, bool permitir_parcial)
    : DataLoader(std::move(entradas), std::move(objetivos), {}, lote, barajar, semilla,
                 permitir_parcial) {
  indices_.resize(static_cast<size_t>(entradas_.Shape().empty() ? 0
                                                               : entradas_.Shape()[0]));
  std::iota(indices_.begin(), indices_.end(), 0);
  if (barajar_) SiguienteEpoca();
}

DataLoader::DataLoader(Tensor entradas, Tensor objetivos, std::vector<int> indices,
                       int lote, bool barajar, uint32_t semilla, bool permitir_parcial)
    : entradas_(std::move(entradas)),
      objetivos_(std::move(objetivos)),
      indices_(std::move(indices)),
      lote_(lote),
      barajar_(barajar),
      permitir_parcial_(permitir_parcial),
      estado_(semilla == 0 ? 1u : semilla) {
  if (lote_ <= 0) throw std::invalid_argument("DataLoader: el lote debe ser positivo.");
  if (entradas_.Shape().empty() || objetivos_.Shape().empty()) {
    throw std::invalid_argument("DataLoader: entradas u objetivos sin ejes.");
  }
  const int n_e = entradas_.Shape()[0];
  const int n_o = objetivos_.Shape()[0];
  if (n_e != n_o) {
    throw std::invalid_argument(
        "DataLoader: " + std::to_string(n_e) + " entradas y " + std::to_string(n_o) +
        " objetivos; no se pueden emparejar.");
  }
  tam_entrada_ = entradas_.TotalSize() / static_cast<size_t>(std::max(n_e, 1));
  tam_objetivo_ = objetivos_.TotalSize() / static_cast<size_t>(std::max(n_o, 1));
}

int DataLoader::NumLotes() const {
  const int n = static_cast<int>(indices_.size());
  if (permitir_parcial_) return (n + lote_ - 1) / lote_;
  return n / lote_;
}

void DataLoader::SiguienteEpoca() {
  if (!barajar_) return;
  // Fisher-Yates con el generador propio: reproducible dada la semilla, y sin
  // tocar nada de fuera.
  for (size_t i = indices_.size(); i > 1; --i) {
    const size_t j = Siguiente(&estado_) % i;
    std::swap(indices_[i - 1], indices_[j]);
  }
}

void DataLoader::Lote(int k, Tensor* x, Tensor* y) const {
  const int total = NumLotes();
  if (k < 0 || k >= total) {
    throw std::out_of_range("DataLoader: lote " + std::to_string(k) + " fuera de [0, " +
                            std::to_string(total) + ").");
  }
  const size_t desde = static_cast<size_t>(k) * lote_;
  const int b = static_cast<int>(std::min(static_cast<size_t>(lote_),
                                          indices_.size() - desde));

  std::vector<int> forma_x = entradas_.Shape();
  forma_x[0] = b;
  std::vector<int> forma_y = objetivos_.Shape();
  forma_y[0] = b;
  x->Resize(forma_x);
  y->Resize(forma_y);

  for (int i = 0; i < b; ++i) {
    const size_t src = static_cast<size_t>(indices_[desde + i]);
    std::memcpy(x->Data() + static_cast<size_t>(i) * tam_entrada_,
                entradas_.Data() + src * tam_entrada_, tam_entrada_ * sizeof(float));
    std::memcpy(y->Data() + static_cast<size_t>(i) * tam_objetivo_,
                objetivos_.Data() + src * tam_objetivo_, tam_objetivo_ * sizeof(float));
  }
}

std::pair<DataLoader, DataLoader> DataLoader::Partir(float fraccion_segunda) const {
  if (fraccion_segunda <= 0.0f || fraccion_segunda >= 1.0f) {
    throw std::invalid_argument("DataLoader::Partir: la fraccion debe estar en (0, 1).");
  }
  const size_t n = indices_.size();
  const size_t n_segunda = static_cast<size_t>(n * fraccion_segunda);
  if (n_segunda == 0 || n_segunda >= n) {
    throw std::invalid_argument("DataLoader::Partir: la fraccion deja una parte vacia.");
  }

  // El corte va sobre los indices YA barajados, asi que las dos partes son
  // muestras del conjunto y no dos tramos contiguos. Cortar el orden original
  // daria particiones sesgadas si los datos vienen agrupados por clase, que es
  // justo como vienen muchos conjuntos.
  std::vector<int> primera(indices_.begin(), indices_.end() - n_segunda);
  std::vector<int> segunda(indices_.end() - n_segunda, indices_.end());

  // `View()` devuelve un tensor que COMPARTE el almacenamiento, y al ser un
  // temporal entra al parametro por valor moviendose, no copiandose. Pasar
  // `entradas_` directamente compilaba igual pero invocaba el constructor de
  // copia, que en `Tensor` reserva memoria nueva: cada hijo se llevaba una copia
  // completa del conjunto. Con MNIST son 180 MB de mas por partir; con algo
  // mayor, una bomba silenciosa.
  return {DataLoader(entradas_.View(entradas_.Shape()), objetivos_.View(objetivos_.Shape()),
                     std::move(primera), lote_, barajar_, estado_, permitir_parcial_),
          DataLoader(entradas_.View(entradas_.Shape()), objetivos_.View(objetivos_.Shape()),
                     std::move(segunda), lote_, barajar_, estado_ + 1u, permitir_parcial_)};
}

}  // namespace data
}  // namespace neuralsuite
