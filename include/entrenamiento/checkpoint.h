// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file checkpoint.h
 * @brief Lo que necesita cualquier entrenamiento largo para poder cortarse.
 *
 * Nacio dentro de `train_diffusion` y se extrajo aqui cuando un segundo
 * entrenador —el del autoencoder del LDM-1— necesito exactamente lo mismo.
 * Copiarlo habrian sido ~150 lineas duplicadas que acabarian divergiendo, y
 * justo en la parte donde un descuido no da error sino un entrenamiento que
 * continua mal.
 *
 * Cada pieza responde a un fallo real que se encontro al construirla:
 *
 *  - `RegistroSellado`: reanudar exigia repetir todas las opciones, y otras
 *    —el total de iteraciones— ni se comprobaban.
 *  - `GuardarCheckpoint`: los archivos de un checkpoint se escribian uno tras
 *    otro, y un corte a mitad los mezclaba sin que nada protestara.
 *  - `GuardarEstadoAdam`: reanudar con los momentos de Adam a cero sacude el
 *    modelo justo al retomar.
 *  - `SemillaIteracion`: sin ella, reanudar no reproducia los mismos lotes.
 */

#ifndef NEURAL_SUITE_INCLUDE_ENTRENAMIENTO_CHECKPOINT_H_
#define NEURAL_SUITE_INCLUDE_ENTRENAMIENTO_CHECKPOINT_H_

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../optimizers.h"
#include "../serialization.h"

namespace neuralsuite {
namespace entrenamiento {

using Metadatos = std::map<std::string, std::string>;

/**
 * @class RegistroSellado
 * @brief Opciones que deciden la trayectoria del entrenamiento y van en el sello.
 *
 * El contrato al reanudar es que **manda el checkpoint**: una opcion sellada que
 * no se pasa en la linea de ordenes se adopta del checkpoint; una que se pasa
 * con otro valor aborta. Pasarla con el mismo valor no hace nada.
 *
 * Los valores se comparan como texto, y los reales con `%.9g`, que representa
 * un `float` sin perdida: dos reales iguales dan el mismo texto.
 */
class RegistroSellado {
 public:
  /** @brief Anota que opciones `--nombre` aparecen en la linea de ordenes. */
  void MarcarDados(int argc, char** argv);
  [[nodiscard]] bool Dado(const std::string& nombre) const { return dados_.count(nombre) > 0; }

  /** @name Registrar una variable del programa bajo un nombre del sello. */
  ///@{
  void Entero(const std::string& nombre, int* variable);
  void Real(const std::string& nombre, float* variable);
  void Texto(const std::string& nombre, std::string* variable);
  ///@}

  /**
   * @brief Adopta del sello lo que no se paso explicitamente.
   *
   * Las claves ausentes del sello se ignoran —checkpoints anteriores—, y las
   * del sello que nadie registro, tambien.
   *
   * @param adoptados recibe `" nombre=valor"` por cada valor adoptado.
   * @return false, con el motivo en `error`, si algo pasado explicitamente
   *         contradice el sello.
   */
  bool Adoptar(const Metadatos& sellado, std::string* adoptados, std::string* error);

  /** @brief El valor actual de todo lo registrado, listo para sellar. */
  [[nodiscard]] Metadatos Valores() const;

 private:
  struct Entrada {
    std::function<std::string()> leer;
    std::function<void(const std::string&)> fijar;
  };
  std::map<std::string, Entrada> entradas_;
  std::set<std::string> dados_;
};

/** @brief Un real como texto sin perdida (`%.9g`). */
std::string TextoReal(double valor);

/**
 * @brief Calentamiento lineal durante `calentamiento` iteraciones y despues
 *        coseno hasta `lr_min` al llegar a `total`.
 *
 * El calentamiento no es cosmetico: Adam arranca con m = v = 0 y la correccion
 * de sesgo hace que los primeros pasos sean del tamano maximo, justo cuando los
 * pesos son aleatorios. `it` empieza en 1.
 */
float TasaAprendizaje(int it, int total, int calentamiento, float lr, float lr_min);

/**
 * @brief Semilla de la iteracion `it`, funcion solo de `(semilla, it)`.
 *
 * Sembrar cada iteracion asi, en vez de arrastrar el estado del generador, es
 * lo que hace que reanudar use los mismos lotes y el mismo ruido que no haber
 * parado.
 */
uint32_t SemillaIteracion(int semilla, int it);

/**
 * @brief Ruta de una copia archivada: el sufijo `_itNNNNNN` va antes de la
 *        extension, con seis cifras para que el orden alfabetico sea el
 *        cronologico.
 */
std::string RutaArchivada(const std::string& base, int it);

/** @brief Un archivo de los que forman un checkpoint. */
struct Parte {
  std::string ruta;
  /** @brief Escribe el archivo en `ruta_temporal`, con `sello` en sus metadatos. */
  std::function<bool(const std::string& ruta_temporal, const Metadatos& sello)> escribir;
};

/**
 * @brief Escribe un checkpoint de varios archivos de forma transaccional.
 *
 * Todos se escriben primero como `.tmp` y solo se mueven a su sitio cuando
 * todos estan completos; si alguno falla se borran los temporales y el
 * checkpoint anterior queda intacto. Todos llevan el mismo `checkpoint_id`
 * —unico por llamada— y la `iteracion`, anadidos a `sello`.
 *
 * Los renombrados no son atomicos como grupo: la ventana baja de segundos a
 * microsegundos, pero no se cierra sin soporte del sistema de archivos. Lo que
 * la cubre es `ComprobarMismoCheckpoint` al cargar.
 */
bool GuardarCheckpoint(const std::vector<Parte>& partes, int iteracion, Metadatos sello,
                       std::string* error);

/**
 * @brief Comprueba que varios archivos pertenecen al mismo checkpoint.
 *
 * Sin esto, archivos de dos checkpoints distintos se cargarian sin protestar:
 * cada uno es valido por separado.
 */
bool ComprobarMismoCheckpoint(const std::vector<Metadatos>& metadatos,
                              const std::vector<std::string>& nombres, std::string* error);

/**
 * @brief Guarda los momentos `m` y `v` de Adam y su contador de pasos.
 *
 * Anade a `meta` las claves `arch = adamw` y `pasos_opt`. Los tensores se llaman
 * `m.i` y `v.i`, en el orden de los parametros.
 */
nsf::Result GuardarEstadoAdam(const std::string& ruta, AdamW& opt, Metadatos meta);

/** @brief Carga lo guardado por `GuardarEstadoAdam` y restaura el contador. */
nsf::Result CargarEstadoAdam(const std::string& ruta, AdamW& opt, Metadatos* meta);

}  // namespace entrenamiento
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_ENTRENAMIENTO_CHECKPOINT_H_
