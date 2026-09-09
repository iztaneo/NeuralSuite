// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file dataloader.h
 * @brief Reparte un conjunto en lotes, con barajado y particion reproducibles.
 */

#ifndef NEURAL_SUITE_INCLUDE_DATA_DATALOADER_H_
#define NEURAL_SUITE_INCLUDE_DATA_DATALOADER_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tensor.h"

namespace neuralsuite {
namespace data {

/**
 * @class DataLoader
 * @brief Entrega lotes de `[B, ...]` a partir de un tensor de ejemplos.
 *
 * Tres cosas que parecen detalles y no lo son:
 *
 * 1. **Generador propio.** El barajado usa su propia semilla y NO toca el
 *    generador global. Si lo tocara, cambiar el tamano de lote alteraria la
 *    inicializacion de los pesos y dos entrenamientos dejarian de ser
 *    comparables por una razon que no tiene nada que ver con lo que se cambio.
 *    Es el mismo cuidado que en la validacion de `train_llm`.
 *
 * 2. **Particion por indices, no por copia.** `Partir()` devuelve dos cargadores
 *    que comparten los mismos datos y se reparten los indices. Copiar los
 *    tensores duplicaria la memoria del conjunto entero para nada.
 *
 * 3. **El ultimo lote incompleto se descarta por defecto**, para mantener el
 *    tamano de lote constante. Lo que varia con un lote a medias no es la escala
 *    del gradiente —`CrossEntropyLoss` promedia, y su backward divide entre
 *    `num_samples`, asi que 17 ejemplos y 32 dan gradientes de la misma escala—
 *    sino el **ruido** del gradiente, el rendimiento, y cualquier estadistica que
 *    dependa del lote. Con una perdida que sumara en vez de promediar si
 *    cambiaria la escala, y por eso conviene no depender de ello.
 *    `permitir_parcial` lo activa para quien lo necesite.
 */
class DataLoader {
 public:
  /**
   * @param entradas `[N, ...]`, un ejemplo por fila del primer eje.
   * @param objetivos `[N]` o `[N, ...]`; debe coincidir en el primer eje.
   */
  DataLoader(Tensor entradas, Tensor objetivos, int lote, bool barajar = true,
             uint32_t semilla = 1234, bool permitir_parcial = false);

  /** @brief Numero de lotes que entrega una epoca. */
  [[nodiscard]] int NumLotes() const;

  /** @brief Ejemplos totales. */
  [[nodiscard]] int Tamano() const { return static_cast<int>(indices_.size()); }

  /**
   * @brief Escribe el lote `k` de la epoca actual en `x` e `y`.
   *
   * `k` va de 0 a `NumLotes() - 1`. Pedir uno fuera de rango lanza excepcion en
   * vez de devolver basura: un bucle mal escrito leeria memoria ajena.
   */
  void Lote(int k, Tensor* x, Tensor* y) const;

  /**
   * @brief Baraja para la epoca siguiente.
   *
   * Hay que llamarlo explicitamente. Que no baraje solo es deliberado: asi dos
   * recorridos sin `SiguienteEpoca()` entregan exactamente lo mismo, que es lo
   * que permite usarlo tambien para evaluar.
   */
  void SiguienteEpoca();

  /** @brief Dos cargadores que se reparten los indices, sin copiar los datos. */
  [[nodiscard]] std::pair<DataLoader, DataLoader> Partir(float fraccion_segunda) const;

  /**
   * @brief Cierto si comparte el almacenamiento del conjunto con `otro`.
   *
   * Existe para que la promesa de `Partir()` sea comprobable y no solo una
   * afirmacion de la documentacion. La primera version SI copiaba: pasaba los
   * tensores como lvalue a un parametro por valor, y el constructor de copia de
   * `Tensor` reserva memoria nueva. Nada fallaba —los datos eran correctos— pero
   * partir un conjunto gastaba el doble de memoria mientras la documentacion
   * decia que no.
   */
  [[nodiscard]] bool CompartioDatosCon(const DataLoader& otro) const {
    return entradas_.SharesStorageWith(otro.entradas_) &&
           objetivos_.SharesStorageWith(otro.objetivos_);
  }

 private:
  DataLoader(Tensor entradas, Tensor objetivos, std::vector<int> indices, int lote,
             bool barajar, uint32_t semilla, bool permitir_parcial);

  Tensor entradas_, objetivos_;
  std::vector<int> indices_;
  int lote_;
  bool barajar_;
  bool permitir_parcial_;
  uint32_t estado_;          // generador propio, independiente del global

  size_t tam_entrada_ = 0;   // elementos por ejemplo
  size_t tam_objetivo_ = 0;
};

}  // namespace data
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DATA_DATALOADER_H_
