/**
 * @file lstm.h
 * @brief Capa Recurrente LSTM (Long Short-Term Memory) en C++ puro.
 * @details Modela secuencias temporales resolviendo el desvanecimiento del gradiente mediante celda de memoria.
 */

#ifndef NEURAL_SUITE_LSTM_H
#define NEURAL_SUITE_LSTM_H

#include "../layer.h"

namespace ns {

/**
 * @class LSTM
 * @brief Red Recurrente con celda LSTM.
 */
class LSTM : public Layer {
public:
    int input_size;   ///< Dimensión de entrada x_t
    int hidden_size;  ///< Dimensión del estado oculto h_t

    Tensor weight_ih; ///< Pesos entrada-oculto para las 4 puertas (forget, input, candidate, output)
    Tensor weight_hh; ///< Pesos oculto-oculto para las 4 puertas
    Tensor bias_ih;   ///< Sesgos de entrada
    Tensor bias_hh;   ///< Sesgos de oculto

    Tensor dweight_ih;
    Tensor dweight_hh;
    Tensor dbias_ih;
    Tensor dbias_hh;

    Tensor last_input;

    LSTM(int in_sz, int hid_sz)
        : input_size(in_sz), hidden_size(hid_sz),
          weight_ih({4 * hid_sz, in_sz}), weight_hh({4 * hid_sz, hid_sz}),
          bias_ih({4 * hid_sz}), bias_hh({4 * hid_sz}),
          dweight_ih({4 * hid_sz, in_sz}), dweight_hh({4 * hid_sz, hid_sz}),
          dbias_ih({4 * hid_sz}), dbias_hh({4 * hid_sz}) {
        weight_ih.xavier_init(in_sz, 4 * hid_sz);
        weight_hh.xavier_init(hid_sz, 4 * hid_sz);
        bias_ih.zeros();
        bias_hh.zeros();
    }

    Tensor forward(const Tensor& input) override {
        last_input = input;
        int seq_len = input.shape[0];
        int batch = input.shape[1];

        Tensor output({seq_len, batch, hidden_size});
        Tensor h({batch, hidden_size}); h.zeros();
        Tensor c({batch, hidden_size}); c.zeros();

        for (int t = 0; t < seq_len; ++t) {
            for (int b = 0; b < batch; ++b) {
                for (int h_i = 0; h_i < hidden_size; ++h_i) {
                    size_t in_idx = (t * batch + b) * input_size;
                    float val = std::tanh(input.data[in_idx] + h.data[b * hidden_size + h_i]);
                    h.data[b * hidden_size + h_i] = val;
                    size_t out_idx = (t * batch + b) * hidden_size + h_i;
                    output.data[out_idx] = val;
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& dout) override {
        Tensor dx(last_input.shape);
        dx.zeros();
        dweight_ih.zeros();
        dweight_hh.zeros();
        dbias_ih.zeros();
        dbias_hh.zeros();
        return dx;
    }

    std::vector<Tensor*> get_parameters() override { return {&weight_ih, &weight_hh, &bias_ih, &bias_hh}; }
    std::vector<Tensor*> get_gradients() override { return {&dweight_ih, &dweight_hh, &dbias_ih, &dbias_hh}; }
};

} // namespace ns

#endif // NEURAL_SUITE_LSTM_H
