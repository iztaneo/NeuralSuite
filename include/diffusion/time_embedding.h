// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file time_embedding.h
 * @brief Convierte el paso de difusion en un vector que la red pueda usar.
 */

#ifndef NEURAL_SUITE_INCLUDE_DIFFUSION_TIME_EMBEDDING_H_
#define NEURAL_SUITE_INCLUDE_DIFFUSION_TIME_EMBEDDING_H_

#include <stdexcept>
#include <string>

#include "../tensor.h"

namespace neuralsuite {
namespace diffusion {

/**
 * @brief Embedding sinusoidal del paso: `[N]` -> `[N, dim]`.
 *
 * La U-Net tiene que saber en que paso del proceso esta, porque quitar ruido
 * cuando queda mucho no se parece a quitarlo cuando queda poco. Pasarle el
 * numero `t` crudo no funciona: un escalar entre 0 y 999 entra en la red con una
 * escala que no se parece a la de sus activaciones, y una sola dimension da muy
 * poca senal para condicionar cientos de canales.
 *
 * La solucion es la misma que la posicion en un transformer: **una escala de
 * frecuencias**. El paso se proyecta sobre senos y cosenos de periodos muy
 * distintos —de 2π a 2π·10000— de modo que pasos cercanos dan vectores
 * parecidos y pasos lejanos, vectores distintos. La red obtiene asi una nocion
 * continua de "cuanto falta" en vez de un indice suelto.
 *
 * La convencion es la de DDPM: la primera mitad de la salida son los senos y la
 * segunda los cosenos, **concatenados, no intercalados**. La variante
 * intercalada existe y es igual de valida, pero mezclarlas produce un embedding
 * que sigue siendo suave y distinto por paso —o sea que parece funcionar— y no
 * coincide con ninguna referencia.
 *
 * `dim` debe ser par: no hay forma de repartir a medias un numero impar de
 * canales entre seno y coseno.
 */
void TimeEmbedding(const Tensor& t, int dim, Tensor* salida, float base = 10000.0f);

}  // namespace diffusion
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_DIFFUSION_TIME_EMBEDDING_H_
