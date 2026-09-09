// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file unet.h
 * @brief La U-Net que predice el ruido, con condicionamiento por paso.
 */

#ifndef NEURAL_SUITE_INCLUDE_DIFFUSION_UNET_H_
#define NEURAL_SUITE_INCLUDE_DIFFUSION_UNET_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../layer.h"
#include "../layers/conv2d.h"
#include "../layers/groupnorm.h"
#include "../layers/linear.h"
#include "../layers/resample2d.h"
#include "resblock.h"

namespace neuralsuite {
namespace diffusion {

/**
 * @class UNet2D
 * @brief `epsilon_theta(x_t, t)`: dado un ruidoso y su paso, predice el ruido.
 *
 * La forma de U viene de bajar la resolucion, procesar en el centro, y volver a
 * subirla concatenando por el camino lo que se guardo al bajar:
 *
 *     [N,1,28,28] ──► entrada ──► res0 ─────────────────────┐ (guardado)
 *                                  │                         │
 *                              bajar 2x                       │
 *                                  ▼                          │
 *                    [N,C1,14,14] res1 ──────────┐ (guardado) │
 *                                  │              │           │
 *                              bajar 2x            │          │
 *                                  ▼               │          │
 *                     [N,C2,7,7]  centro           │          │
 *                                  │               │          │
 *                              subir 2x            │          │
 *                                  ▼               │          │
 *                            concat ◄──────────────┘          │
 *                                  │                          │
 *                             res_subida1                     │
 *                              subir 2x                       │
 *                                  ▼                          │
 *                            concat ◄─────────────────────────┘
 *                                  │
 *                             res_subida0 ──► norma ──► salida ──► [N,1,28,28]
 *
 * Los **saltos** son lo que la hace funcionar. Al bajar se pierde detalle fino;
 * concatenar el mapa de alta resolucion que se guardo devuelve esa informacion
 * al subir, en vez de tener que reconstruirla. Por eso hizo falta `Concat` con
 * su derivada antes de llegar aqui: sin ella la union cortaria el grafo.
 *
 * Es deliberadamente pequena. El objetivo del escalon 4 no es una U-Net buena
 * sino una que se pueda **sobreajustar a proposito** con un punado de imagenes:
 * si no puede memorizar ocho digitos, hay un defecto en la arquitectura, en el
 * backward o en el entrenamiento, y ninguna cantidad de epocas lo va a arreglar.
 */
class UNet2D : public Module {
 public:
  /**
   * @param canales resolucion base; los niveles usan `canales` y `2*canales`.
   * @param dim_t dimension del embedding del paso.
   */
  UNet2D(int canales_imagen = 1, int canales = 32, int dim_t = 64, int grupos = 8);

  /** @brief `[N, C, H, W]` con el paso `[N]` -> ruido predicho, misma forma. */
  Tensor Forward(const Tensor& x, const Tensor& pasos);

  /** @brief Propaga y devuelve el gradiente respecto a `x`. */
  Tensor Backward(const Tensor& dout);

  /**
   * @name Acceso a los submodulos
   *
   * Existen para la paridad: permiten inyectar los pesos de la referencia pieza
   * a pieza y por nombre. Cargarlos por la lista plana de `GetParameters()`
   * tambien funcionaria, pero entonces un fallo no distinguiria entre un
   * cableado malo y un orden mal adivinado, que es justo lo que la paridad
   * tiene que separar aqui.
   */
  ///@{
  Conv2D& ConvEntrada() { return conv_entrada_; }
  Conv2D& ConvSalida() { return conv_salida_; }
  GroupNormLayer& NormSalida() { return norm_salida_; }
  ResBlockTiempo& ResBaja0() { return res_baja0_; }
  ResBlockTiempo& ResBaja1() { return res_baja1_; }
  ResBlockTiempo& ResCentro() { return res_centro_; }
  ResBlockTiempo& ResAlta1() { return res_alta1_; }
  ResBlockTiempo& ResAlta0() { return res_alta0_; }
  ///@}

  /**
   * @name Pesos y gradientes
   *
   * Se derivan de `Parameters()`, que sale de los `Register()` del constructor.
   * Antes eran dos listas escritas a mano en paralelo —exactamente lo que el
   * docstring de `Layer` advierte—: anadir un submodulo obligaba a acordarse de
   * tocar las dos, y olvidar una habria dejado un peso sin optimizar sin que
   * nada fallase.
   */
  ///@{
  [[nodiscard]] std::vector<Tensor*> GetParameters();
  [[nodiscard]] std::vector<Tensor*> GetGradients();
  [[nodiscard]] size_t NumParametros();
  ///@}

  /**
   * @name Persistencia
   *
   * Sin esto un entrenamiento de horas no deja nada en disco. El formato es el
   * mismo NSF del resto del proyecto: los pesos van con su ruta en el arbol
   * (`res_baja0.conv1.weight`), asi que cargar un archivo de otra arquitectura
   * falla al comprobar los nombres en vez de leer numeros del tamano correcto
   * y producir un modelo silenciosamente equivocado.
   */
  ///@{
  /**
   * @param extra metadatos adicionales que se escriben en el archivo junto a
   *        los de la arquitectura. Sirve para sellar un checkpoint con un
   *        identificador comun a los varios archivos que lo forman.
   */
  bool GuardarPesos(const std::string& ruta,
                    const std::map<std::string, std::string>& extra = {});

  /**
   * @param leidos si se da, recibe todos los metadatos del archivo, incluidos
   *        los que la arquitectura no exige.
   */
  bool CargarPesos(const std::string& ruta,
                   std::map<std::string, std::string>* leidos = nullptr);

  /** @brief Lo que el archivo declara esperar: canales, dim del tiempo, grupos. */
  [[nodiscard]] std::map<std::string, std::string> MetadatosArquitectura() const;
  ///@}

 private:
  int canales_imagen_, canales_, dim_t_, grupos_;

  Conv2D conv_entrada_;
  ResBlockTiempo res_baja0_, res_baja1_, res_centro_, res_alta1_, res_alta0_;
  Downsample2D bajar0_, bajar1_;
  Upsample2D subir1_, subir0_;
  GroupNormLayer norm_salida_;
  Conv2D conv_salida_;

  // Lo guardado al bajar, que se concatena al subir.
  Tensor salto0_, salto1_, t_emb_;
  Tensor pre_salida_, act_salida_;
  int n_ = 0, h_ = 0, w_ = 0;
};

}  // namespace diffusion
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DIFFUSION_UNET_H_
