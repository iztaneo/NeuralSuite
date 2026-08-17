// Copyright 2026 NeuralSuite Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

/**
 * @file tensor.h
 * @brief Multi-dimensional Tensor & Linear Algebra Engine strictly following the Google C++ Style Guide.
 * @see https://google.github.io/styleguide/cppguide.html
 */

#ifndef NEURAL_SUITE_INCLUDE_TENSOR_H_
#define NEURAL_SUITE_INCLUDE_TENSOR_H_

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuralsuite {

/**
 * @class Tensor
 * @brief Multi-dimensional array storing floating point numbers in contiguous memory.
 * @details Conforms to Google C++ Style Guide: RAII memory management, explicit constructors,
 * const-correctness, and `data_` / `shape_` private members.
 */
class Tensor {
 public:
  Tensor();
  explicit Tensor(const std::vector<int>& dims);
  Tensor(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  ~Tensor();

  Tensor& operator=(const Tensor& other);
  Tensor& operator=(Tensor&& other) noexcept;

  // Accessors
  [[nodiscard]] size_t TotalSize() const;
  [[nodiscard]] const std::vector<int>& Shape() const { return shape_; }
  [[nodiscard]] float* Data() { return data_; }
  [[nodiscard]] const float* Data() const { return data_; }

  void Reshape(const std::vector<int>& new_shape);
  void Fill(float val);
  void Zeros();
  void Ones();
  void RandomNormal(float mean = 0.0f, float stddev = 0.02f);
  void RandomUniform(float min_val = -0.1f, float max_val = 0.1f);
  void XavierInit(int fan_in, int fan_out);

  // Element access operators
  inline float& operator[](size_t idx) { return data_[idx]; }
  inline const float& operator[](size_t idx) const { return data_[idx]; }

  void PrintSummary(const std::string& name = "") const;

 private:
  float* data_ = nullptr;
  std::vector<int> shape_;
};

// ============================================================================
// LINEAR ALGEBRA & MATH PRIMITIVES (GOOGLE C++ STYLE GUIDE)
// ============================================================================

/**
 * @brief General Matrix Multiplication (GEMM): C = A * B
 */
void MatMul(const Tensor& A, const Tensor& B, Tensor& C);

/**
 * @brief 2D Matrix Transposition: C = A^T
 */
Tensor Transpose(const Tensor& A);

/**
 * @brief Pre-LN Layer Normalization Forward Pass
 */
void LayerNormForward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                      Tensor& out, Tensor& mean, Tensor& rstd, float eps = 1e-5f);

/**
 * @brief Layer Normalization Backward Pass
 */
void LayerNormBackward(const Tensor& dout, const Tensor& x, const Tensor& gamma,
                       const Tensor& mean, const Tensor& rstd, Tensor& dx,
                       Tensor& dgamma, Tensor& dbeta);

/**
 * @brief Numerically Stable Softmax
 */
void SoftmaxForward(const Tensor& input, Tensor& output);

/**
 * @brief Causal Masked Softmax for Transformer Attention
 */
void CausalSoftmaxForward(const Tensor& input, Tensor& output, int seq_len);

/**
 * @brief Gaussian Error Linear Unit (GELU) Forward Pass
 */
void GeluForward(const Tensor& input, Tensor& output);

/**
 * @brief GELU Backward Pass
 */
void GeluBackward(const Tensor& dout, const Tensor& input, Tensor& dx);

/**
 * @brief ReLU Forward & Backward
 */
void ReluForward(const Tensor& input, Tensor& output);
void ReluBackward(const Tensor& dout, const Tensor& input, Tensor& dx);

/**
 * @brief Sigmoid Forward & Backward
 */
void SigmoidForward(const Tensor& input, Tensor& output);
void SigmoidBackward(const Tensor& dout, const Tensor& output, Tensor& dx);

/**
 * @brief Tanh Forward & Backward
 */
void TanhForward(const Tensor& input, Tensor& output);
void TanhBackward(const Tensor& dout, const Tensor& output, Tensor& dx);

/**
 * @brief Elementwise Addition, Subtraction, Multiplication
 */
void ElementwiseAdd(const Tensor& a, const Tensor& b, Tensor& out);
void ElementwiseSub(const Tensor& a, const Tensor& b, Tensor& out);
void ElementwiseMul(const Tensor& a, const Tensor& b, Tensor& out);

// Alias para compatibilidad ns::
namespace ns = neuralsuite;

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_TENSOR_H_
