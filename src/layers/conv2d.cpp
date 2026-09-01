// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de layers/conv2d.h. La cabecera se queda con la interfaz y el
// porque del diseno; aqui va el porque de cada linea.

#include "layers/conv2d.h"

namespace neuralsuite {

Tensor Conv2DReference::Forward(const Tensor& input) {
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

Tensor Conv2DReference::Backward(const Tensor& dout) {
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

Tensor Conv2D::Forward(const Tensor& input) {
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

    // Cada imagen del lote es independiente: no hay reduccion, asi que el
    // resultado es identico bit a bit al de un solo hilo.
    parallel::ParallelFor(batch, /*min_per_thread=*/1, [&](int desde, int hasta) {
      Tensor columnas({k, posiciones});
      Tensor parcial;
      for (int b = desde; b < hasta; ++b) {
        Im2Col(input, b, height, width, out_h, out_w, &columnas);
        MatMul(pesos, columnas, parcial);
        std::memcpy(output.Data() + static_cast<size_t>(b) * out_channels_ * posiciones,
                    parcial.Data(),
                    static_cast<size_t>(out_channels_) * posiciones * sizeof(float));
      }
    });

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

Tensor Conv2D::Backward(const Tensor& dout) {
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

    // Un hueco por imagen para las contribuciones a los gradientes. Sumarlas
    // aqui mismo desde varios hilos seria una carrera, y hacerlo con bloqueos
    // dejaria el resultado a merced del orden de llegada.
    const size_t tam_pesos = static_cast<size_t>(out_channels_) * k;
    std::vector<float> dpesos_por_imagen(tam_pesos * batch, 0.0f);
    std::vector<float> dsesgo_por_imagen(static_cast<size_t>(out_channels_) * batch, 0.0f);

    parallel::ParallelFor(batch, /*min_per_thread=*/1, [&](int desde, int hasta) {
      Tensor columnas({k, posiciones});
      Tensor dpesos_lote, dcolumnas;
      Tensor dout_2d({out_channels_, posiciones});

      for (int b = desde; b < hasta; ++b) {
        // dout de esta imagen, leido como [canales_salida, posiciones].
        std::memcpy(dout_2d.Data(),
                    dout.Data() + static_cast<size_t>(b) * out_channels_ * posiciones,
                    static_cast<size_t>(out_channels_) * posiciones * sizeof(float));

        // dW = dout * columnas^T
        Im2Col(last_input_, b, height, width, out_h, out_w, &columnas);
        const Tensor columnas_t = Transpose(columnas);   // [posiciones, k]
        MatMul(dout_2d, columnas_t, dpesos_lote);
        std::memcpy(dpesos_por_imagen.data() + tam_pesos * b, dpesos_lote.Data(),
                    tam_pesos * sizeof(float));

        // dbias = suma de dout sobre las posiciones.
        for (int oc = 0; oc < out_channels_; ++oc) {
          const float* fila = dout_2d.Data() + static_cast<size_t>(oc) * posiciones;
          float suma = 0.0f;
          for (int i = 0; i < posiciones; ++i) suma += fila[i];
          dsesgo_por_imagen[static_cast<size_t>(out_channels_) * b + oc] = suma;
        }

        // dColumnas = W^T * dout, y de ahi se reparte a la imagen. Cada imagen
        // escribe en su propia franja de dx, asi que no hay carrera.
        MatMul(pesos_t, dout_2d, dcolumnas);
        Col2Im(dcolumnas, b, height, width, out_h, out_w, &dx);
      }
    });

    // La reduccion, en orden de imagen y en un solo hilo: asi el gradiente sale
    // igual con uno o con diez.
    for (int b = 0; b < batch; ++b) {
      const float* origen = dpesos_por_imagen.data() + tam_pesos * b;
      for (size_t i = 0; i < tam_pesos; ++i) dweight[i] += origen[i];
      for (int oc = 0; oc < out_channels_; ++oc) {
        dbias[oc] += dsesgo_por_imagen[static_cast<size_t>(out_channels_) * b + oc];
      }
    }
    return dx;
  }

void Conv2D::Im2Col(const Tensor& entrada, int b, int height, int width, int out_h, int out_w, Tensor* salida) const {
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

void Conv2D::Col2Im(const Tensor& columnas, int b, int height, int width, int out_h, int out_w, Tensor* destino) const {
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

}  // namespace neuralsuite
