/**
 * @file tensor.h
 * @brief Motor de Matemáticas y Álgebra Lineal en C++17 puro para NeuralSuite.
 * @details Proporciona la estructura de Tensor multidimensional contiguo en Heap con gestión de memoria RAII,
 * multiplicaciones de matrices GEMM (General Matrix Multiplication), normalización de capa (LayerNorm),
 * activación GELU (Gaussian Error Linear Unit), Softmax Causal, ReLU, Sigmoide, Tanh y gradientes analíticos.
 *
 * @author NeuralSuite Core Team
 * @date 2026
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
 * @brief Arreglo multidimensional de números flotantes (float) en memoria contigua Heap.
 * @details Implementa la semántica RAII (Resource Acquisition Is Initialization) y movimiento de C++11/17.
 */
class Tensor {
public:
    float* data;               ///< Puntero a la memoria contigua de datos flotantes
    std::vector<int> shape;    ///< Vector que almacena las dimensiones del tensor (ej: [batch, seq, dim])

    /** @brief Constructor por defecto: inicializa un tensor vacío */
    Tensor();

    /** 
     * @brief Constructor con dimensiones específicas
     * @param dims Vector con los tamaños de cada dimensión (ej: {32, 128})
     */
    Tensor(const std::vector<int>& dims);

    /** @brief Constructor de copia (Deep Copy) */
    Tensor(const Tensor& other);

    /** @brief Constructor de movimiento (Move Semantics C++11) */
    Tensor(Tensor&& other) noexcept;

    /** @brief Destructor: libera la memoria Heap ocupada por `data` */
    ~Tensor();

    /** @brief Operador de asignación por copia */
    Tensor& operator=(const Tensor& other);

    /** @brief Operador de asignación por movimiento */
    Tensor& operator=(Tensor&& other) noexcept;

    /** 
     * @brief Calcula el número total de elementos contenidos en el tensor
     * @return Multiplicación de todas las dimensiones en `shape`
     */
    size_t total_size() const;

    /** 
     * @brief Cambia la forma del tensor sin modificar los datos subyacentes
     * @param new_shape Nuevas dimensiones esperadas
     */
    void reshape(const std::vector<int>& new_shape);

    /** @brief Rellena todos los elementos con un valor constante */
    void fill(float val);
    
    /** @brief Inicializa todos los elementos a cero (0.0f) */
    void zeros();

    /** @brief Inicializa todos los elementos a uno (1.0f) */
    void ones();

    /** 
     * @brief Inicialización estocástica Normal Gaussiana N(mean, std^2)
     * @param mean Media de la distribución (por defecto 0.0)
     * @param std Desviación estándar (por defecto 0.02)
     */
    void random_normal(float mean = 0.0f, float std = 0.02f);

    /** 
     * @brief Inicialización estocástica Uniforme U(min, max)
     */
    void random_uniform(float min_val = -0.1f, float max_val = 0.1f);

    /** 
     * @brief Inicialización Xavier / Glorot para capas densas (Linear)
     * @details Mantiene la varianza de los gradientes constante entre capas: U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
     */
    void xavier_init(int fan_in, int fan_out);

    /** @brief Acceso rápido por índice plano (L-value) */
    inline float& operator[](size_t idx) { return data[idx]; }

    /** @brief Acceso rápido por índice plano (Const L-value) */
    inline const float& operator[](size_t idx) const { return data[idx]; }
    
    /** @brief Imprime un resumen de dimensiones y tamaño en consola */
    void print_summary(const std::string& name = "") const;
};

// ============================================================================
// FUNCIONES Y PRIMITIVAS DE ÁLGEBRA LINEAL (MOTOR MATEMÁTICO EN C++)
// ============================================================================

/**
 * @brief Multiplicación de Matrices 2D Generalizada (GEMM): C = A * B
 * @details Calcula la multiplicación matricial dividiendo el trabajo entre hilos mediante OpenMP.
 * Fórmulas: C[i, j] = sum_{k=0}^{K-1} ( A[i, k] * B[k, j] )
 * @param A Matriz M x K
 * @param B Matriz K x N
 * @param C Matriz de salida asignada de tamaño M x N
 * @note Complejidad Temporal: O(M * N * K)
 */
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief Transposición de Matriz 2D: C = A^T
 * @param A Matriz de entrada de tamaño M x N
 * @return Nueva matriz transpuesta de tamaño N x M
 */
Tensor transpose(const Tensor& A);

/**
 * @brief Paso Forward de Normalización de Capa (Pre-LN LayerNorm)
 * @details Normaliza las características en la dimensión interna d:
 * \hat{x}_i = (x_i - mean) / sqrt(var + eps)
 * out_i = \hat{x}_i * gamma_i + beta_i
 * @param x Tensor de entrada de forma [N, D]
 * @param gamma Vector de escala entrenable [D]
 * @param beta Vector de desplazamiento entrenable [D]
 * @param out Tensor de salida normalizado [N, D]
 * @param mean Caché de medias por muestra [N]
 * @param rstd Caché de inverso de desviación estándar 1/sqrt(var+eps) [N]
 * @param eps Valor constante para evitar división por cero (1e-5)
 */
void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, Tensor& out, Tensor& mean, Tensor& rstd, float eps = 1e-5f);

/**
 * @brief Paso Backward de Normalización de Capa (LayerNorm Gradient)
 * @details Calcula analíticamente dx, dgamma y dbeta mediante la regla de la cadena.
 */
void layernorm_backward(const Tensor& dout, const Tensor& x, const Tensor& gamma, const Tensor& mean, const Tensor& rstd, Tensor& dx, Tensor& dgamma, Tensor& dbeta);

/**
 * @brief Softmax Estable numéricamente
 * @details Aplica Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))) para evitar overflow.
 */
void softmax_forward(const Tensor& input, Tensor& output);

/**
 * @brief Softmax Causal con Máscara Triangular Inferior
 * @details Aplica la máscara causal estableciendo las posiciones futuras j > i a 0.0.
 */
void causal_softmax_forward(const Tensor& input, Tensor& output, int seq_len);

/**
 * @brief Activación GELU (Gaussian Error Linear Unit) - Paso Forward
 * @details Aproximación analítica de GPT-2: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void gelu_forward(const Tensor& input, Tensor& output);

/**
 * @brief Activación GELU - Paso Backward (Gradiente Analítico)
 */
void gelu_backward(const Tensor& dout, const Tensor& input, Tensor& dx);

/** @brief Activación ReLU Forward: max(0, x) */
void relu_forward(const Tensor& input, Tensor& output);
/** @brief Activación ReLU Backward */
void relu_backward(const Tensor& dout, const Tensor& input, Tensor& dx);

/** @brief Activación Sigmoide Forward: 1 / (1 + exp(-x)) */
void sigmoid_forward(const Tensor& input, Tensor& output);
/** @brief Activación Sigmoide Backward */
void sigmoid_backward(const Tensor& dout, const Tensor& output, Tensor& dx);

/** @brief Activación Tanh Forward: tanh(x) */
void tanh_forward(const Tensor& input, Tensor& output);
/** @brief Activación Tanh Backward */
void tanh_backward(const Tensor& dout, const Tensor& output, Tensor& dx);

/** @brief Operaciones elementales entre tensores (Suma, Resta, Multiplicación Hadamard) */
void elementwise_add(const Tensor& a, const Tensor& b, Tensor& out);
void elementwise_sub(const Tensor& a, const Tensor& b, Tensor& out);
void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace ns

#endif // NEURAL_SUITE_TENSOR_H
