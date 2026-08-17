/**
 * @file test_suite.cpp
 * @brief Suite de Pruebas Unitarias Automáticas y Verificación Numérica de Gradientes para NeuralSuite en C++.
 */

#include "tensor.h"
#include "tokenizer.h"
#include "activations.h"
#include "losses.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_matmul() {
    std::cout << "🧪 [Test 1] Multiplicación de Matrices (GEMM)... " << std::flush;
    ns::Tensor A({2, 3});
    ns::Tensor B({3, 2});
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A.data[0] = 1; A.data[1] = 2; A.data[2] = 3;
    A.data[3] = 4; A.data[4] = 5; A.data[5] = 6;

    // B = [[7, 8], [9, 1], [2, 3]]
    B.data[0] = 7; B.data[1] = 8;
    B.data[2] = 9; B.data[3] = 1;
    B.data[4] = 2; B.data[5] = 3;

    ns::Tensor C;
    ns::matmul(A, B, C);

    // C = [[31, 19], [85, 55]]
    assert(std::abs(C.data[0] - 31.0f) < 1e-4f);
    assert(std::abs(C.data[1] - 19.0f) < 1e-4f);
    assert(std::abs(C.data[2] - 85.0f) < 1e-4f);
    assert(std::abs(C.data[3] - 55.0f) < 1e-4f);

    std::cout << "PASADO ✅\n" << std::flush;
}

void test_layernorm() {
    std::cout << "🧪 [Test 2] Normalización de Capa (LayerNorm)... " << std::flush;
    ns::Tensor x({1, 4});
    x.data[0] = 2.0f; x.data[1] = 4.0f; x.data[2] = 4.0f; x.data[3] = 6.0f;

    ns::Tensor gamma({4}); gamma.ones();
    ns::Tensor beta({4}); beta.zeros();

    ns::Tensor out, mean, rstd;
    ns::layernorm_forward(x, gamma, beta, out, mean, rstd);

    // Media = (2+4+4+6)/4 = 4.0
    assert(std::abs(mean.data[0] - 4.0f) < 1e-4f);

    std::cout << "PASADO ✅\n" << std::flush;
}

void test_tokenizer() {
    std::cout << "🧪 [Test 3] Tokenizador de Caracteres C++... " << std::flush;
    std::string sample = "Hello C++!";
    ns::CharTokenizer tok(sample);
    
    std::vector<int> encoded = tok.encode(sample);
    std::string decoded = tok.decode(encoded);

    assert(sample == decoded);
    std::cout << "PASADO ✅\n" << std::flush;
}

void test_gradient_check_gelu() {
    std::cout << "🧪 [Test 4] Verificación de Gradiente GELU por Diferencias Finitas... " << std::flush;
    ns::Tensor x({1, 1});
    x.data[0] = 1.5f;

    ns::Tensor dout({1, 1});
    dout.data[0] = 1.0f;

    ns::Tensor dx;
    ns::gelu_backward(dout, x, dx);

    // Gradiente numérico por diferencias finitas: (f(x+eps) - f(x-eps)) / (2*eps)
    float eps = 1e-4f;
    ns::Tensor x_plus({1, 1}), x_minus({1, 1});
    x_plus.data[0] = x.data[0] + eps;
    x_minus.data[0] = x.data[0] - eps;

    ns::Tensor y_plus, y_minus;
    ns::gelu_forward(x_plus, y_plus);
    ns::gelu_forward(x_minus, y_minus);

    float num_grad = (y_plus.data[0] - y_minus.data[0]) / (2.0f * eps);
    float diff = std::abs(dx.data[0] - num_grad);

    assert(diff < 1e-3f);
    std::cout << "PASADO ✅ (Diff: " << diff << ")\n" << std::flush;
}

int main() {
    std::cout << "============================================================\n" << std::flush;
    std::cout << "🚀 Ejecutando Suite de Pruebas Unitarias de NeuralSuite (C++)\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;

    test_matmul();
    test_layernorm();
    test_tokenizer();
    test_gradient_check_gelu();

    std::cout << "============================================================\n" << std::flush;
    std::cout << "✅ ¡Todas las pruebas unitarias pasaron con éxito!\n" << std::flush;
    std::cout << "============================================================\n" << std::flush;
    return 0;
}
