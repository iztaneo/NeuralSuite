// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

#include "tensor.h"
#include "parallel.h"

namespace neuralsuite {

static std::mt19937 g_rng(1337);

void ManualSeed(uint32_t seed) { g_rng.seed(seed); }

namespace {

// Un Tensor nunca debe poder existir en un estado internamente inválido, así
// que la forma se valida en los únicos puntos donde se asigna: el constructor,
// Reshape() y Resize(). Con eso TotalSize() puede seguir siendo una simple
// multiplicación sin comprobaciones en el camino caliente.
void ValidateShape(const std::vector<int>& dims) {
  size_t total = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    const int d = dims[i];
    if (d < 0) {
      throw std::invalid_argument(
          "Tensor: dimension negativa (" + std::to_string(d) + ") en el eje " +
          std::to_string(i) + ".");
    }
    if (d == 0) {
      total = 0;
      continue;
    }
    if (total != 0 && total > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
      throw std::length_error("Tensor: el numero total de elementos desborda size_t.");
    }
    total *= static_cast<size_t>(d);
  }
}

}  // namespace

Tensor::Tensor() : shape_({}) {}

Tensor::Tensor(const std::vector<int>& dims) : shape_(dims) {
  ValidateShape(shape_);
  const size_t sz = TotalSize();
  if (sz > 0) storage_ = std::make_shared<Storage>(sz, 0.0f);
}

// La copia es profunda aunque el original sea una vista: el codigo que guarda
// instantaneas (last_input_ = input) necesita quedarse con datos propios, no
// con un alias que cambie bajo sus pies.
Tensor::Tensor(const Tensor& other) : shape_(other.shape_) {
  const size_t sz = other.TotalSize();
  if (sz > 0) {
    storage_ = std::make_shared<Storage>(sz);
    std::memcpy(storage_->data(), other.Data(), sz * sizeof(float));
  }
}

Tensor::Tensor(Tensor&& other) noexcept
    : storage_(std::move(other.storage_)),
      offset_(other.offset_),
      shape_(std::move(other.shape_)) {
  other.offset_ = 0;
  other.shape_.clear();
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    // Se construye el nuevo almacenamiento antes de soltar el anterior: si la
    // reserva falla, el objeto conserva intactos sus datos y su forma.
    const size_t sz = other.TotalSize();
    std::shared_ptr<Storage> new_storage;
    if (sz > 0) {
      new_storage = std::make_shared<Storage>(sz);
      std::memcpy(new_storage->data(), other.Data(), sz * sizeof(float));
    }
    storage_ = std::move(new_storage);
    offset_ = 0;
    shape_ = other.shape_;
  }
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    storage_ = std::move(other.storage_);
    offset_ = other.offset_;
    shape_ = std::move(other.shape_);
    other.offset_ = 0;
    other.shape_.clear();
  }
  return *this;
}

Tensor Tensor::View(const std::vector<int>& new_shape) const {
  ValidateShape(new_shape);

  size_t requested = new_shape.empty() ? 0 : 1;
  for (int d : new_shape) requested *= static_cast<size_t>(d);

  const size_t current = TotalSize();
  if (requested != current) {
    throw std::invalid_argument(
        "View: la forma pedida tiene " + std::to_string(requested) +
        " elementos y el tensor tiene " + std::to_string(current) + ".");
  }

  Tensor v;
  v.storage_ = storage_;   // se comparte, no se copia
  v.offset_ = offset_;
  v.shape_ = new_shape;
  return v;
}

size_t Tensor::TotalSize() const {
  if (shape_.empty()) return 0;
  size_t sz = 1;
  for (int d : shape_) sz *= d;
  return sz;
}

void Tensor::Reshape(const std::vector<int>& new_shape) {
  ValidateShape(new_shape);

  const size_t current = TotalSize();
  size_t requested = 1;
  for (int d : new_shape) requested *= static_cast<size_t>(d);
  if (new_shape.empty()) requested = 0;

  // Reshape reinterpreta; nunca reasigna ni descarta datos. Cambiar el número
  // de elementos es una operación distinta y tiene su propio nombre.
  if (requested != current) {
    throw std::invalid_argument(
        "Reshape: la nueva forma tiene " + std::to_string(requested) +
        " elementos y el tensor tiene " + std::to_string(current) +
        ". Usa Resize() si la intencion es reasignar memoria.");
  }
  shape_ = new_shape;
}

void Tensor::Resize(const std::vector<int>& new_shape) {
  ValidateShape(new_shape);
  const size_t old_sz = TotalSize();
  shape_ = new_shape;
  const size_t new_sz = TotalSize();

  if (new_sz != old_sz) {
    // Se reserva antes de soltar el bloque anterior, y siempre uno propio: un
    // Resize sobre una vista debe dejar de compartir memoria, no reescribir la
    // del tensor del que salio.
    auto new_storage = (new_sz > 0) ? std::make_shared<Storage>(new_sz, 0.0f) : nullptr;
    storage_ = std::move(new_storage);
    offset_ = 0;
  }
}

void Tensor::Fill(float val) {
  size_t sz = TotalSize();
  float* d = Data();
  for (size_t i = 0; i < sz; ++i) d[i] = val;
}

void Tensor::Zeros() { Fill(0.0f); }
void Tensor::Ones() { Fill(1.0f); }

void Tensor::RandomNormal(float mean, float stddev) {
  std::normal_distribution<float> dist(mean, stddev);
  size_t sz = TotalSize();
  float* d = Data();
  for (size_t i = 0; i < sz; ++i) d[i] = dist(g_rng);
}

void Tensor::RandomUniform(float min_val, float max_val) {
  std::uniform_real_distribution<float> dist(min_val, max_val);
  size_t sz = TotalSize();
  float* d = Data();
  for (size_t i = 0; i < sz; ++i) d[i] = dist(g_rng);
}

void Tensor::XavierInit(int fan_in, int fan_out) {
  float limit = std::sqrt(6.0f / (fan_in + fan_out));
  RandomUniform(-limit, limit);
}

void Tensor::PrintSummary(const std::string& name) const {
  std::cout << "Tensor [" << name << "] Shape: (";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
  }
  std::cout << ") Total size: " << TotalSize() << "\n";
}

// ============================================================================
// LINEAR ALGEBRA & MATH PRIMITIVES IMPLEMENTATION
// ============================================================================

void MatMul(const Tensor& A, const Tensor& B, Tensor& C) {
  // Sin estas comprobaciones, un tensor con el rango equivocado accede a
  // Shape()[1] fuera del vector y luego recorre memoria ajena al leer A y B.
  if (A.Shape().size() != 2 || B.Shape().size() != 2) {
    throw std::invalid_argument("MatMul: ambos operandos deben ser de rango 2.");
  }
  if (A.Shape()[1] != B.Shape()[0]) {
    throw std::invalid_argument(
        "MatMul: dimensiones incompatibles, A es [" + std::to_string(A.Shape()[0]) + ", " +
        std::to_string(A.Shape()[1]) + "] y B es [" + std::to_string(B.Shape()[0]) + ", " +
        std::to_string(B.Shape()[1]) + "]; A.cols debe igualar B.rows.");
  }

  int M = A.Shape()[0];
  int K = A.Shape()[1];
  int N = B.Shape()[1];

  C.Resize({M, N});
  C.Zeros();

  // Cada hilo se queda con un bloque de filas de C, que nadie mas escribe: sin
  // reduccion no cambia el orden de las sumas, de modo que el resultado es
  // identico bit a bit al de un solo hilo. Esa propiedad es la que permite
  // paralelizar sin tocar la comparacion contra PyTorch.
  parallel::ParallelFor(M, /*min_per_thread=*/16, [&](int row_begin, int row_end) {
    // Se calculan cuatro filas de C a la vez. El bucle anterior leia una fila
    // entera de B para producir una sola fila de C, de modo que cada elemento
    // de B viajaba de memoria a registro para una unica multiplicacion; asi
    // sirve para cuatro. Medido en un solo hilo, sube de 35 a 44 GFLOP/s.
    //
    // Cuatro y no mas: con seis u ocho acumuladores el compilador se queda sin
    // registros vectoriales y tiene que volcarlos a memoria, con lo que el
    // rendimiento cae por debajo del punto de partida (medido: 41 y 33).
    //
    // Para un mismo elemento de C, las contribuciones se siguen sumando en
    // orden de k creciente, igual que antes, de modo que el resultado es
    // identico bit a bit.
    int i = row_begin;
    for (; i + 4 <= row_end; i += 4) {
      for (int k = 0; k < K; ++k) {
        const float a0 = A[(i + 0) * K + k];
        const float a1 = A[(i + 1) * K + k];
        const float a2 = A[(i + 2) * K + k];
        const float a3 = A[(i + 3) * K + k];
        const float* b_row = &B[k * N];
        float* c0 = &C[(i + 0) * N];
        float* c1 = &C[(i + 1) * N];
        float* c2 = &C[(i + 2) * N];
        float* c3 = &C[(i + 3) * N];
        // El bucle interno recorre memoria contigua y los compiladores lo
        // autovectorizan; `omp simd` solo lo aceptaba MSVC con una bandera
        // experimental.
        for (int j = 0; j < N; ++j) {
          const float b = b_row[j];
          c0[j] += a0 * b;
          c1[j] += a1 * b;
          c2[j] += a2 * b;
          c3[j] += a3 * b;
        }
      }
    }
    // Las filas que no completan un grupo de cuatro.
    for (; i < row_end; ++i) {
      for (int k = 0; k < K; ++k) {
        const float a = A[i * K + k];
        const float* b_row = &B[k * N];
        float* c_row = &C[i * N];
        for (int j = 0; j < N; ++j) {
          c_row[j] += a * b_row[j];
        }
      }
    }
  });
}



Tensor Transpose(const Tensor& A) {
  int M = A.Shape()[0];
  int N = A.Shape()[1];
  Tensor C({N, M});

  parallel::ParallelFor(M, /*min_per_thread=*/32, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = 0; j < N; ++j) {
        C[j * M + i] = A[i * N + j];
      }
    }
  });
  return C;
}

void LayerNormForward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                      Tensor& out, Tensor& mean, Tensor& rstd, float eps) {
  int num_dims = x.Shape().size();
  int D = x.Shape()[num_dims - 1];
  int N = x.TotalSize() / D;

  out.Resize(x.Shape());
  mean.Resize({N});
  rstd.Resize({N});

  parallel::ParallelFor(N, /*min_per_thread=*/16, [&](int row_begin, int row_end) {
  for (int i = row_begin; i < row_end; ++i) {
    float m = 0.0f;
    for (int j = 0; j < D; ++j) m += x[i * D + j];
    m /= D;
    mean[i] = m;

    float v = 0.0f;
    for (int j = 0; j < D; ++j) {
      float diff = x[i * D + j] - m;
      v += diff * diff;
    }
    v /= D;
    float rs = 1.0f / std::sqrt(v + eps);
    rstd[i] = rs;

    for (int j = 0; j < D; ++j) {
      float x_hat = (x[i * D + j] - m) * rs;
      out[i * D + j] = x_hat * gamma[j] + beta[j];
    }
  }
  });
}

void LayerNormBackward(const Tensor& dout, const Tensor& x, const Tensor& gamma,
                       const Tensor& mean, const Tensor& rstd, Tensor& dx,
                       Tensor& dgamma, Tensor& dbeta) {
  int num_dims = x.Shape().size();
  int D = x.Shape()[num_dims - 1];
  int N = x.TotalSize() / D;

  dx.Resize(x.Shape());
  dgamma.Resize({D}); dgamma.Zeros();
  dbeta.Resize({D}); dbeta.Zeros();

  for (int i = 0; i < N; ++i) {
    float m = mean[i];
    float rs = rstd[i];

    float sum_dout = 0.0f;
    float sum_dout_xhat = 0.0f;

    for (int j = 0; j < D; ++j) {
      float x_hat = (x[i * D + j] - m) * rs;
      float d = dout[i * D + j];
      dgamma[j] += d * x_hat;
      dbeta[j] += d;

      float d_xhat = d * gamma[j];
      sum_dout += d_xhat;
      sum_dout_xhat += d_xhat * x_hat;
    }

    for (int j = 0; j < D; ++j) {
      float x_hat = (x[i * D + j] - m) * rs;
      float d_xhat = dout[i * D + j] * gamma[j];
      dx[i * D + j] = (rs / D) * (D * d_xhat - sum_dout - x_hat * sum_dout_xhat);
    }
  }
}


void SoftmaxForward(const Tensor& input, Tensor& output) {
  // Opera sobre el ultimo eje, sea cual sea el rango. Antes exigia rango 2 y
  // leia Shape()[1] directamente, de modo que un tensor de rango 3 accedia a un
  // eje que no le correspondia.
  if (input.Shape().empty()) {
    throw std::invalid_argument("SoftmaxForward: el tensor no tiene ejes.");
  }
  const int D = input.Shape().back();
  const int N = static_cast<int>(input.TotalSize()) / D;
  output.Resize(input.Shape());

  parallel::ParallelFor(N, /*min_per_thread=*/16, [&](int row_begin, int row_end) {
  for (int i = row_begin; i < row_end; ++i) {
    float max_val = input[i * D];
    for (int j = 1; j < D; ++j) {
      if (input[i * D + j] > max_val) max_val = input[i * D + j];
    }

    float sum = 0.0f;
    for (int j = 0; j < D; ++j) {
      float e = std::exp(input[i * D + j] - max_val);
      output[i * D + j] = e;
      sum += e;
    }

    for (int j = 0; j < D; ++j) {
      output[i * D + j] /= sum;
    }
  }
  });
}

void CausalSoftmaxForward(const Tensor& input, Tensor& output, int seq_len) {
  int num_dims = input.Shape().size();
  int B = (num_dims == 3) ? input.Shape()[0] : 1;

  output.Resize(input.Shape());

  parallel::ParallelFor(B, /*min_per_thread=*/1, [&](int b_begin, int b_end) {
  for (int b = b_begin; b < b_end; ++b) {

    for (int i = 0; i < seq_len; ++i) {
      int offset = (b * seq_len + i) * seq_len;
      float max_val = input[offset];
      for (int j = 1; j <= i; ++j) {
        if (input[offset + j] > max_val) max_val = input[offset + j];
      }

      float sum = 0.0f;
      for (int j = 0; j <= i; ++j) {
        float e = std::exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum += e;
      }

      for (int j = 0; j <= i; ++j) {
        output[offset + j] /= sum;
      }
      for (int j = i + 1; j < seq_len; ++j) {
        output[offset + j] = 0.0f;
      }
    }
  }
  });
}

void GeluForward(const Tensor& input, Tensor& output) {
  size_t sz = input.TotalSize();
  output.Resize(input.Shape());

  // El indice va con signo: el OpenMP de MSVC (2.0) rechaza size_t en el bucle
  // de un `parallel for`.
  parallel::ParallelFor(static_cast<int>(sz), /*min_per_thread=*/4096,
                        [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const float x = input[i];
      const float cube = 0.044715f * x * x * x;
      const float inner = 0.7978845608f * (x + cube);
      output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
  });
}

void GeluBackward(const Tensor& dout, const Tensor& input, Tensor& dx) {
  size_t sz = input.TotalSize();
  dx.Resize(input.Shape());

  parallel::ParallelFor(static_cast<int>(sz), /*min_per_thread=*/4096,
                        [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const float x = input[i];
      const float cube = 0.044715f * x * x * x;
      const float inner = 0.7978845608f * (x + cube);
      const float th = std::tanh(inner);
      const float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
      const float d_gelu = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * d_inner;
      dx[i] = dout[i] * d_gelu;
    }
  });
}

// Las operaciones elementales de este bloque reparten su bucle entre hilos.
// Escribian una posicion de salida por cada posicion de entrada, sin ninguna
// reduccion, asi que el reparto no cambia el orden de ninguna suma y el
// resultado es identico bit a bit al de un solo hilo.
//
// Estaban sin repartir por omision, no por decision: GELU y softmax si lo
// hacian desde el principio. Medido sobre un paso del CRNN de OCR, ReLU y
// MaxPool2D juntos eran 15.2 ms de 28.2 -el 53.8%- una vez que la convolucion
// y la celda recurrente dejaron de dominar.
void ReluForward(const Tensor& input, Tensor& output) {
  const int sz = static_cast<int>(input.TotalSize());
  output.Resize(input.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
  });
}

void ReluBackward(const Tensor& dout, const Tensor& input, Tensor& dx) {
  const int sz = static_cast<int>(input.TotalSize());
  dx.Resize(input.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      dx[i] = (input[i] > 0.0f) ? dout[i] : 0.0f;
    }
  });
}

void SigmoidForward(const Tensor& input, Tensor& output) {
  const int sz = static_cast<int>(input.TotalSize());
  output.Resize(input.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
  });
}

void SigmoidBackward(const Tensor& dout, const Tensor& output, Tensor& dx) {
  const int sz = static_cast<int>(output.TotalSize());
  dx.Resize(output.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      float s = output[i];
      dx[i] = dout[i] * s * (1.0f - s);
    }
  });
}

void TanhForward(const Tensor& input, Tensor& output) {
  const int sz = static_cast<int>(input.TotalSize());
  output.Resize(input.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      output[i] = std::tanh(input[i]);
    }
  });
}

void TanhBackward(const Tensor& dout, const Tensor& output, Tensor& dx) {
  const int sz = static_cast<int>(output.TotalSize());
  dx.Resize(output.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) {
      float th = output[i];
      dx[i] = dout[i] * (1.0f - th * th);
    }
  });
}

void ElementwiseAdd(const Tensor& a, const Tensor& b, Tensor& out) {
  const int sz = static_cast<int>(a.TotalSize());
  out.Resize(a.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) out[i] = a[i] + b[i];
  });
}

void ElementwiseSub(const Tensor& a, const Tensor& b, Tensor& out) {
  const int sz = static_cast<int>(a.TotalSize());
  out.Resize(a.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) out[i] = a[i] - b[i];
  });
}

void ElementwiseMul(const Tensor& a, const Tensor& b, Tensor& out) {
  const int sz = static_cast<int>(a.TotalSize());
  out.Resize(a.Shape());
  parallel::ParallelFor(sz, /*min_per_thread=*/4096, [&](int desde, int hasta) {
    for (int i = desde; i < hasta; ++i) out[i] = a[i] * b[i];
  });
}

}  // namespace neuralsuite
