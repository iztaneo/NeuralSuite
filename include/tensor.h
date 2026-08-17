/**
 * @file tensor.h
 * @brief Motor de Matemáticas y Álgebra Lineal en C++17 puro para NeuralSuite.
 * @details Proporciona asignación dinámica RAII, multiplicación de matrices (GEMM),
 * LayerNorm, Softmax Causal, GELU, ReLU y operaciones elementales paralelizadas.
 */

#ifndef NEURAL_SUITE_TENSOR_H
#define NEURAL_SUITE_TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace ns {

/**
 * @class Tensor
 * @brief Estructura de Tensor multidimensional contiguo en memoria Heap.
 */
class Tensor {
public:
    float* data;
    std::vector<int> shape;

    Tensor();
    Tensor(const std::vector<int>& dims);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    size_t total_size() const;
    void reshape(const std::vector<int>& new_shape);
    void fill(float val);
    
    // Métodos de inicialización estocástica de pesos
    void zeros();
    void ones();
    void random_normal(float mean = 0.0f, float std = 0.02f);
    void random_uniform(float min_val = -0.1f, float max_val = 0.1f);
    void xavier_init(int fan_in, int fan_out);

    // Acceso a elementos
    inline float& operator[](size_t idx) { return data[idx]; }
    inline const float& operator[](size_t idx) const { return data[idx]; }
    
    void print_summary(const std::string& name = "") const;
};

// ============================================================================
// PRIMITIVAS DE ÁLGEBRA LINEAL Y OPERACIONES MATEMÁTICAS (GEMM, LAYER NORM, ETC.)
// ============================================================================

/**
 * @brief Multiplicación de Matrices 2D Generalizada (GEMM): C = A * B
 * @param A Matriz M x K
 * @param B Matriz K x N
 * @param C Matriz de Salida M x N
 */
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Transposición de Matriz 2D: C = A^T
 */
Tensor transpose(const Tensor& A);

/**
 * @brief Normalización de Capa (Pre-LN LayerNorm)
 */
void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, Tensor& out, Tensor& mean, Tensor& rstd, float eps = 1e-5f);
void layernorm_backward(const Tensor& dout, const Tensor& x, const Tensor& gamma, const Tensor& mean, const Tensor& rstd, Tensor& dx, Tensor& dgamma, Tensor& dbeta);

/**
 * @brief Softmax Causal para la atención del Transformer
 */
void softmax_forward(const Tensor& input, Tensor& output);
void causal_softmax_forward(const Tensor& input, Tensor& output, int seq_len);

/**
 * @brief Activación GELU (Gaussian Error Linear Unit)
 */
void gelu_forward(const Tensor& input, Tensor& output);
void gelu_backward(const Tensor& dout, const Tensor& input, Tensor& dx);

/**
 * @brief Activaciones elementales: ReLU, Sigmoid, Tanh
 */
void relu_forward(const Tensor& input, Tensor& output);
void relu_backward(const Tensor& dout, const Tensor& input, Tensor& dx);

void sigmoid_forward(const Tensor& input, Tensor& output);
void sigmoid_backward(const Tensor& dout, const Tensor& output, Tensor& dx);

void tanh_forward(const Tensor& input, Tensor& output);
void tanh_backward(const Tensor& dout, const Tensor& output, Tensor& dx);

/**
 * @brief Operaciones elementales entre tensores
 */
void elementwise_add(const Tensor& a, const Tensor& b, Tensor& out);
void elementwise_sub(const Tensor& a, const Tensor& b, Tensor& out);
void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace ns

#endif // NEURAL_SUITE_TENSOR_H
