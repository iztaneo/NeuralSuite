/**
 * @file conv2d.h
 * @brief Capa Convolucional 2D para procesamiento de imágenes y visión por computador.
 */

#ifndef NEURAL_SUITE_CONV2D_H
#define NEURAL_SUITE_CONV2D_H

#include "../layer.h"

namespace ns {

/**
 * @class Conv2D
 * @brief Operación Convolucional bidimensional.
 * @details Desliza filtros de dimensión (out_channels, in_channels, k, k) sobre tensores 4D [Batch, Channels, Height, Width].
 */
class Conv2D : public Layer {
public:
    int in_channels;   ///< Número de canales de entrada
    int out_channels;  ///< Número de mapas de características de salida (filtros)
    int kernel_size;   ///< Tamaño del filtro convolucional k x k
    int stride;        ///< Paso de deslizamiento del filtro
    int padding;       ///< Relleno de ceros en los bordes de la imagen

    Tensor weight;     ///< Pesos de los filtros [out_channels, in_channels, k, k]
    Tensor bias;       ///< Vector de sesgos [out_channels]
    Tensor dweight;    ///< Acumulador de gradientes de filtros
    Tensor dbias;      ///< Acumulador de gradientes de sesgos
    Tensor last_input; ///< Caché del tensor de entrada

    /**
     * @brief Constructor de la Capa Convolucional 2D.
     */
    Conv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0)
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(str), padding(pad),
          weight({out_ch, in_ch, k_size, k_size}), bias({out_ch}),
          dweight({out_ch, in_ch, k_size, k_size}), dbias({out_ch}) {
        weight.xavier_init(in_ch * k_size * k_size, out_ch * k_size * k_size);
        bias.zeros();
    }

    /**
     * @brief Computa la convolución 2D: Y = X * K + b
     */
    Tensor forward(const Tensor& input) override {
        last_input = input;
        int B = input.shape[0];
        int H = input.shape[2];
        int W = input.shape[3];

        int out_h = (H + 2 * padding - kernel_size) / stride + 1;
        int out_w = (W + 2 * padding - kernel_size) / stride + 1;

        Tensor output({B, out_channels, out_h, out_w});

        for (int b = 0; b < B; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float val = bias.data[oc];
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        size_t in_idx = ((b * in_channels + ic) * H + ih) * W + iw;
                                        size_t w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                        val += input.data[in_idx] * weight.data[w_idx];
                                    }
                                }
                            }
                        }
                        size_t out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                        output.data[out_idx] = val;
                    }
                }
            }
        }
        return output;
    }

    /**
     * @brief Retropropagación de gradientes en Convolución 2D
     */
    Tensor backward(const Tensor& dout) override {
        int B = last_input.shape[0];
        int H = last_input.shape[2];
        int W = last_input.shape[3];

        int out_h = dout.shape[2];
        int out_w = dout.shape[3];

        Tensor dx({B, in_channels, H, W});
        dx.zeros();
        dweight.zeros();
        dbias.zeros();

        for (int b = 0; b < B; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        size_t dout_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                        float d = dout.data[dout_idx];
                        dbias.data[oc] += d;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        size_t in_idx = ((b * in_channels + ic) * H + ih) * W + iw;
                                        size_t w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                        
                                        dweight.data[w_idx] += d * last_input.data[in_idx];
                                        dx.data[in_idx] += d * weight.data[w_idx];
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

    std::vector<Tensor*> get_parameters() override { return {&weight, &bias}; }
    std::vector<Tensor*> get_gradients() override { return {&dweight, &dbias}; }
};

} // namespace ns

#endif // NEURAL_SUITE_CONV2D_H
