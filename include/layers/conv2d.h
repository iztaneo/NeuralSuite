// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file conv2d.h
 * @brief 2D Convolutional Layer following Google C++ Style Guide.
 */

#ifndef NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
#define NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "../layer.h"
#include "../parameter.h"

namespace neuralsuite {

/**
 * @class Conv2DReference
 * @brief Convolucion escrita como su definicion: siete bucles anidados.
 *
 * Se conserva a proposito. Es lenta —1.5 GFLOP/s frente a los 232 que alcanza
 * `MatMul`— pero se lee al lado de la formula y no tiene nada que pueda
 * desalinearse: ni reordenaciones de memoria, ni indices calculados. `Conv2D`,
 * que si los tiene, se comprueba contra esta.
 *
 * Que el oraculo viva en el mismo archivo y se compile siempre es deliberado.
 * Una version de referencia que haya que desenterrar de un commit antiguo deja
 * de usarse a la primera prisa.
 */
class Conv2DReference : public Layer {
 public:
  Conv2DReference(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
      : in_channels_(in_ch),
        out_channels_(out_ch),
        kernel_size_(k_size),
        stride_(str),
        padding_(pad),
        weight_({out_ch, in_ch, k_size, k_size}),
        bias_({out_ch}) {
    Register(&weight_, "weight");
    Register(&bias_, "bias");
    weight_.Value().XavierInit(in_ch * k_size * k_size, out_ch * k_size * k_size);
    bias_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    last_input_ = input;
    int batch_size = input.Shape()[0];
    int height = input.Shape()[2];
    int width = input.Shape()[3];

    int out_h = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_w = (width + 2 * padding_ - kernel_size_) / stride_ + 1;

    Tensor output({batch_size, out_channels_, out_h, out_w});

    for (int b = 0; b < batch_size; ++b) {
      for (int oc = 0; oc < out_channels_; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            float val = bias_.Value()[oc];
            for (int ic = 0; ic < in_channels_; ++ic) {
              for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                  int ih = oh * stride_ + kh - padding_;
                  int iw = ow * stride_ + kw - padding_;
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    size_t in_idx = ((b * in_channels_ + ic) * height + ih) * width + iw;
                    size_t w_idx = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;
                    val += input[in_idx] * weight_.Value()[w_idx];
                  }
                }
              }
            }
            size_t out_idx = ((b * out_channels_ + oc) * out_h + oh) * out_w + ow;
            output[out_idx] = val;
          }
        }
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    int batch_size = last_input_.Shape()[0];
    int height = last_input_.Shape()[2];
    int width = last_input_.Shape()[3];

    int out_h = dout.Shape()[2];
    int out_w = dout.Shape()[3];

    Tensor dx({batch_size, in_channels_, height, width});
    dx.Zeros();
    Tensor& dweight = weight_.Grad();
    Tensor& dbias = bias_.Grad();
    dweight.Zeros();
    dbias.Zeros();

    for (int b = 0; b < batch_size; ++b) {
      for (int oc = 0; oc < out_channels_; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            size_t dout_idx = ((b * out_channels_ + oc) * out_h + oh) * out_w + ow;
            float d = dout[dout_idx];
            dbias[oc] += d;

            for (int ic = 0; ic < in_channels_; ++ic) {
              for (int kh = 0; kh < kernel_size_; ++kh) {
                for (int kw = 0; kw < kernel_size_; ++kw) {
                  int ih = oh * stride_ + kh - padding_;
                  int iw = ow * stride_ + kw - padding_;
                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    size_t in_idx = ((b * in_channels_ + ic) * height + ih) * width + iw;
                    size_t w_idx = ((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_ + kw;

                    dweight[w_idx] += d * last_input_[in_idx];
                    dx[in_idx] += d * weight_.Value()[w_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
    return dx;
  }

 private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  Parameter weight_;
  Parameter bias_;
  Tensor last_input_;
};


/**
 * @class Conv2D
 * @brief Convolucion resuelta como una multiplicacion de matrices.
 *
 * La definicion de la convolucion son siete bucles anidados, y escrita asi
 * alcanzaba 1.5 GFLOP/s. Perfilando un paso de entrenamiento del CRNN de OCR,
 * las tres convoluciones se llevaban el 85.7% del tiempo: 514 ms la de 16 a 32
 * canales, 410 ms la de 32 a 64. Y corrian en un solo hilo teniendo diez, sin
 * tocar el pool que ya existia.
 *
 * La reformulacion es la clasica, `im2col`: cada ventana de la imagen se copia
 * como una columna de una matriz, y entonces la convolucion entera es
 *
 *     salida[canal, posicion] = pesos[canal, k] * columnas[k, posicion]
 *
 * con `k = canales_entrada * tamano * tamano`. Eso es exactamente un `MatMul`,
 * que en este proyecto ya esta paralelizado y con bloqueo de registros. Se
 * cambia trabajo por memoria: las columnas repiten cada pixel tantas veces como
 * ventanas lo cubren, `k` veces en el peor caso.
 *
 * El orden de los ejes no es casual. Las columnas se construyen ya
 * transpuestas, `[k, posiciones]`, para que los pesos entren en la
 * multiplicacion con la forma en que estan guardados —`[canales, k]`— y la
 * salida salga directamente como `[canales, posiciones]`, que es la disposicion
 * del tensor de salida. Cualquier otro orden obliga a transponer el resultado
 * de cada lote.
 *
 * No da los mismos bits que `Conv2DReference`: la suma se hace en otro orden y
 * en punto flotante eso importa. Lo que si debe cumplir es coincidir dentro del
 * redondeo, y de eso se encarga la prueba que las contrasta.
 */
class Conv2D : public Layer {
 public:
  Conv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
      : in_channels_(in_ch),
        out_channels_(out_ch),
        kernel_size_(k_size),
        stride_(str),
        padding_(pad),
        weight_({out_ch, in_ch, k_size, k_size}),
        bias_({out_ch}) {
    if (in_ch <= 0 || out_ch <= 0 || k_size <= 0 || str <= 0 || pad < 0) {
      throw std::invalid_argument("Conv2D: parametros de la convolucion no validos.");
    }
    Register(&weight_, "weight");
    Register(&bias_, "bias");
    weight_.Value().XavierInit(in_ch * k_size * k_size, out_ch * k_size * k_size);
    bias_.Value().Zeros();
  }

  Tensor Forward(const Tensor& input) override {
    if (input.Shape().size() != 4) {
      throw std::invalid_argument("Conv2D: la entrada debe ser [lote, canales, alto, ancho].");
    }
    if (input.Shape()[1] != in_channels_) {
      throw std::invalid_argument(
          "Conv2D: la entrada tiene " + std::to_string(input.Shape()[1]) +
          " canales y la capa espera " + std::to_string(in_channels_) + ".");
    }
    last_input_ = input;
    const int batch = input.Shape()[0];
    const int height = input.Shape()[2];
    const int width = input.Shape()[3];
    const int out_h = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
    const int out_w = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
    if (out_h <= 0 || out_w <= 0) {
      throw std::invalid_argument("Conv2D: el nucleo no cabe en la entrada.");
    }

    const int k = in_channels_ * kernel_size_ * kernel_size_;
    const int posiciones = out_h * out_w;

    // Los pesos ya estan como [canales_salida, k]: solo hay que releerlos.
    const Tensor pesos = weight_.Value().View({out_channels_, k});

    Tensor output({batch, out_channels_, out_h, out_w});
    Tensor columnas({k, posiciones});
    Tensor parcial;

    for (int b = 0; b < batch; ++b) {
      Im2Col(input, b, height, width, out_h, out_w, &columnas);
      MatMul(pesos, columnas, parcial);
      std::memcpy(output.Data() + static_cast<size_t>(b) * out_channels_ * posiciones,
                  parcial.Data(), static_cast<size_t>(out_channels_) * posiciones * sizeof(float));
    }

    // El sesgo se suma al final: meterlo en la matriz obligaria a anadir una
    // fila de unos a las columnas, y eso cuesta mas memoria que este bucle.
    for (int b = 0; b < batch; ++b) {
      for (int oc = 0; oc < out_channels_; ++oc) {
        float* fila = output.Data() + (static_cast<size_t>(b) * out_channels_ + oc) * posiciones;
        const float sesgo = bias_.Value()[oc];
        for (int i = 0; i < posiciones; ++i) fila[i] += sesgo;
      }
    }
    return output;
  }

  Tensor Backward(const Tensor& dout) override {
    const int batch = last_input_.Shape()[0];
    const int height = last_input_.Shape()[2];
    const int width = last_input_.Shape()[3];
    const int out_h = dout.Shape()[2];
    const int out_w = dout.Shape()[3];

    const int k = in_channels_ * kernel_size_ * kernel_size_;
    const int posiciones = out_h * out_w;

    Tensor dx({batch, in_channels_, height, width});
    dx.Zeros();
    Tensor& dweight = weight_.Grad();
    Tensor& dbias = bias_.Grad();
    dweight.Zeros();
    dbias.Zeros();

    const Tensor pesos = weight_.Value().View({out_channels_, k});
    const Tensor pesos_t = Transpose(pesos);   // [k, canales_salida]

    Tensor columnas({k, posiciones});
    Tensor dpesos_lote, dcolumnas;

    for (int b = 0; b < batch; ++b) {
      // dout de este lote, leido como [canales_salida, posiciones].
      Tensor dout_2d({out_channels_, posiciones});
      std::memcpy(dout_2d.Data(),
                  dout.Data() + static_cast<size_t>(b) * out_channels_ * posiciones,
                  static_cast<size_t>(out_channels_) * posiciones * sizeof(float));

      // dW = dout * columnas^T, acumulado sobre el lote.
      Im2Col(last_input_, b, height, width, out_h, out_w, &columnas);
      const Tensor columnas_t = Transpose(columnas);   // [posiciones, k]
      MatMul(dout_2d, columnas_t, dpesos_lote);
      for (size_t i = 0; i < dweight.TotalSize(); ++i) dweight[i] += dpesos_lote[i];

      // dbias = suma de dout sobre las posiciones.
      for (int oc = 0; oc < out_channels_; ++oc) {
        const float* fila = dout_2d.Data() + static_cast<size_t>(oc) * posiciones;
        float suma = 0.0f;
        for (int i = 0; i < posiciones; ++i) suma += fila[i];
        dbias[oc] += suma;
      }

      // dColumnas = W^T * dout, y de ahi se reparte a la imagen.
      MatMul(pesos_t, dout_2d, dcolumnas);
      Col2Im(dcolumnas, b, height, width, out_h, out_w, &dx);
    }
    return dx;
  }

  [[nodiscard]] const Tensor& Weight() const { return weight_.Value(); }
  Tensor& Weight() { return weight_.Value(); }
  [[nodiscard]] const Tensor& Bias() const { return bias_.Value(); }
  Tensor& Bias() { return bias_.Value(); }

 private:
  /**
   * @brief Copia cada ventana de la imagen `b` como una columna.
   *
   * `salida` queda como `[canales*tamano*tamano, posiciones]`. Los pixeles que
   * caen fuera por el relleno se dejan a cero, que es lo que significa rellenar.
   */
  void Im2Col(const Tensor& entrada, int b, int height, int width, int out_h, int out_w,
              Tensor* salida) const {
    const int posiciones = out_h * out_w;
    float* dst = salida->Data();
    const float* src = entrada.Data() + static_cast<size_t>(b) * in_channels_ * height * width;

    for (int ic = 0; ic < in_channels_; ++ic) {
      for (int kh = 0; kh < kernel_size_; ++kh) {
        for (int kw = 0; kw < kernel_size_; ++kw) {
          const int fila = (ic * kernel_size_ + kh) * kernel_size_ + kw;
          float* destino = dst + static_cast<size_t>(fila) * posiciones;
          for (int oh = 0; oh < out_h; ++oh) {
            const int ih = oh * stride_ + kh - padding_;
            if (ih < 0 || ih >= height) {
              std::memset(destino + static_cast<size_t>(oh) * out_w, 0, out_w * sizeof(float));
              continue;
            }
            const float* origen = src + (static_cast<size_t>(ic) * height + ih) * width;
            float* linea = destino + static_cast<size_t>(oh) * out_w;
            for (int ow = 0; ow < out_w; ++ow) {
              const int iw = ow * stride_ + kw - padding_;
              linea[ow] = (iw >= 0 && iw < width) ? origen[iw] : 0.0f;
            }
          }
        }
      }
    }
  }

  /** @brief La inversa de Im2Col: devuelve cada columna a su sitio, sumando. */
  void Col2Im(const Tensor& columnas, int b, int height, int width, int out_h, int out_w,
              Tensor* destino) const {
    const int posiciones = out_h * out_w;
    float* dst = destino->Data() + static_cast<size_t>(b) * in_channels_ * height * width;

    for (int ic = 0; ic < in_channels_; ++ic) {
      for (int kh = 0; kh < kernel_size_; ++kh) {
        for (int kw = 0; kw < kernel_size_; ++kw) {
          const int fila = (ic * kernel_size_ + kh) * kernel_size_ + kw;
          const float* origen = columnas.Data() + static_cast<size_t>(fila) * posiciones;
          for (int oh = 0; oh < out_h; ++oh) {
            const int ih = oh * stride_ + kh - padding_;
            if (ih < 0 || ih >= height) continue;
            float* linea = dst + (static_cast<size_t>(ic) * height + ih) * width;
            const float* fuente = origen + static_cast<size_t>(oh) * out_w;
            for (int ow = 0; ow < out_w; ++ow) {
              const int iw = ow * stride_ + kw - padding_;
              // Un mismo pixel lo tocan varias ventanas: sus contribuciones se
              // suman, y por eso esto no puede ser una copia.
              if (iw >= 0 && iw < width) linea[iw] += fuente[ow];
            }
          }
        }
      }
    }
  }

  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  Parameter weight_;
  Parameter bias_;
  Tensor last_input_;
};

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_LAYERS_CONV2D_H_
