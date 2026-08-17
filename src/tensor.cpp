#include "tensor.h"

namespace ns {

// Generator aleatorio estático global
static std::mt19937 g_rng(1337);

Tensor::Tensor() : data(nullptr), shape({}) {}

Tensor::Tensor(const std::vector<int>& dims) : shape(dims) {
    size_t sz = total_size();
    data = (sz > 0) ? new float[sz]() : nullptr;
}

Tensor::Tensor(const Tensor& other) : shape(other.shape) {
    size_t sz = total_size();
    if (sz > 0) {
        data = new float[sz];
        std::memcpy(data, other.data, sz * sizeof(float));
    } else {
        data = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data), shape(std::move(other.shape)) {
    other.data = nullptr;
}

Tensor::~Tensor() {
    delete[] data;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;
        shape = other.shape;
        size_t sz = total_size();
        if (sz > 0) {
            data = new float[sz];
            std::memcpy(data, other.data, sz * sizeof(float));
        } else {
            data = nullptr;
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data = other.data;
        shape = std::move(other.shape);
        other.data = nullptr;
    }
    return *this;
}

size_t Tensor::total_size() const {
    if (shape.empty()) return 0;
    size_t sz = 1;
    for (int d : shape) sz *= d;
    return sz;
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t old_sz = total_size();
    shape = new_shape;
    size_t new_sz = total_size();

    if (new_sz != old_sz) {
        delete[] data;
        data = (new_sz > 0) ? new float[new_sz]() : nullptr;
    }
}


void Tensor::fill(float val) {
    size_t sz = total_size();
    for (size_t i = 0; i < sz; ++i) data[i] = val;
}

void Tensor::zeros() { fill(0.0f); }
void Tensor::ones() { fill(1.0f); }

void Tensor::random_normal(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    size_t sz = total_size();
    for (size_t i = 0; i < sz; ++i) data[i] = dist(g_rng);
}

void Tensor::random_uniform(float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    size_t sz = total_size();
    for (size_t i = 0; i < sz; ++i) data[i] = dist(g_rng);
}

void Tensor::xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    random_uniform(-limit, limit);
}

void Tensor::print_summary(const std::string& name) const {
    std::cout << "Tensor [" << name << "] Shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << ") Total size: " << total_size() << "\n";
}

// ============================================================================
// IMPLEMENTACIÓN DE PRIMITIVAS MATEMÁTICAS EN C++
// ============================================================================

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    C.reshape({M, N});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

Tensor transpose(const Tensor& A) {
    int M = A.shape[0];
    int N = A.shape[1];
    Tensor C({N, M});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C.data[j * M + i] = A.data[i * N + j];
        }
    }
    return C;
}

void layernorm_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta, Tensor& out, Tensor& mean, Tensor& rstd, float eps) {
    int N = x.shape[0]; // Batch / Seq len
    int D = x.shape[1]; // Embedding dimension

    out.reshape({N, D});
    mean.reshape({N});
    rstd.reshape({N});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        float m = 0.0f;
        for (int j = 0; j < D; ++j) m += x.data[i * D + j];
        m /= D;
        mean.data[i] = m;

        float v = 0.0f;
        for (int j = 0; j < D; ++j) {
            float diff = x.data[i * D + j] - m;
            v += diff * diff;
        }
        v /= D;
        float rs = 1.0f / std::sqrt(v + eps);
        rstd.data[i] = rs;

        for (int j = 0; j < D; ++j) {
            float x_hat = (x.data[i * D + j] - m) * rs;
            out.data[i * D + j] = x_hat * gamma.data[j] + beta.data[j];
        }
    }
}

void layernorm_backward(const Tensor& dout, const Tensor& x, const Tensor& gamma, const Tensor& mean, const Tensor& rstd, Tensor& dx, Tensor& dgamma, Tensor& dbeta) {
    int N = x.shape[0];
    int D = x.shape[1];

    dx.reshape({N, D});
    dgamma.reshape({D}); dgamma.zeros();
    dbeta.reshape({D}); dbeta.zeros();

    for (int i = 0; i < N; ++i) {
        float m = mean.data[i];
        float rs = rstd.data[i];

        float sum_dout = 0.0f;
        float sum_dout_xhat = 0.0f;

        for (int j = 0; j < D; ++j) {
            float x_hat = (x.data[i * D + j] - m) * rs;
            float d = dout.data[i * D + j];
            dgamma.data[j] += d * x_hat;
            dbeta.data[j] += d;

            float d_xhat = d * gamma.data[j];
            sum_dout += d_xhat;
            sum_dout_xhat += d_xhat * x_hat;
        }

        for (int j = 0; j < D; ++j) {
            float x_hat = (x.data[i * D + j] - m) * rs;
            float d_xhat = dout.data[i * D + j] * gamma.data[j];
            dx.data[i * D + j] = (rs / D) * (D * d_xhat - sum_dout - x_hat * sum_dout_xhat);
        }
    }
}

void softmax_forward(const Tensor& input, Tensor& output) {
    int N = input.shape[0];
    int D = input.shape[1];
    output.reshape({N, D});

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        float max_val = input.data[i * D];
        for (int j = 1; j < D; ++j) {
            if (input.data[i * D + j] > max_val) max_val = input.data[i * D + j];
        }

        float sum = 0.0f;
        for (int j = 0; j < D; ++j) {
            float e = std::exp(input.data[i * D + j] - max_val);
            output.data[i * D + j] = e;
            sum += e;
        }

        for (int j = 0; j < D; ++j) {
            output.data[i * D + j] /= sum;
        }
    }
}

void causal_softmax_forward(const Tensor& input, Tensor& output, int seq_len) {
    int B = input.shape[0]; // Batch or Head count
    output.reshape({B, seq_len, seq_len});

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            int offset = (b * seq_len + i) * seq_len;
            float max_val = input.data[offset];
            for (int j = 1; j <= i; ++j) {
                if (input.data[offset + j] > max_val) max_val = input.data[offset + j];
            }

            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float e = std::exp(input.data[offset + j] - max_val);
                output.data[offset + j] = e;
                sum += e;
            }

            for (int j = 0; j <= i; ++j) {
                output.data[offset + j] /= sum;
            }
            for (int j = i + 1; j < seq_len; ++j) {
                output.data[offset + j] = 0.0f; // Máscara Causal (Futuro = 0)
            }
        }
    }
}

void gelu_forward(const Tensor& input, Tensor& output) {
    size_t sz = input.total_size();
    output.reshape(input.shape);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sz; ++i) {
        float x = input.data[i];
        float cube = 0.044715f * x * x * x;
        float inner = 0.7978845608f * (x + cube); // sqrt(2 / pi)
        output.data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void gelu_backward(const Tensor& dout, const Tensor& input, Tensor& dx) {
    size_t sz = input.total_size();
    dx.reshape(input.shape);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sz; ++i) {
        float x = input.data[i];
        float cube = 0.044715f * x * x * x;
        float inner = 0.7978845608f * (x + cube);
        float th = std::tanh(inner);
        float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
        float d_gelu = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * d_inner;
        dx.data[i] = dout.data[i] * d_gelu;
    }
}

void relu_forward(const Tensor& input, Tensor& output) {
    size_t sz = input.total_size();
    output.reshape(input.shape);
    for (size_t i = 0; i < sz; ++i) {
        output.data[i] = (input.data[i] > 0.0f) ? input.data[i] : 0.0f;
    }
}

void relu_backward(const Tensor& dout, const Tensor& input, Tensor& dx) {
    size_t sz = input.total_size();
    dx.reshape(input.shape);
    for (size_t i = 0; i < sz; ++i) {
        dx.data[i] = (input.data[i] > 0.0f) ? dout.data[i] : 0.0f;
    }
}

void sigmoid_forward(const Tensor& input, Tensor& output) {
    size_t sz = input.total_size();
    output.reshape(input.shape);
    for (size_t i = 0; i < sz; ++i) {
        output.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
    }
}

void sigmoid_backward(const Tensor& dout, const Tensor& output, Tensor& dx) {
    size_t sz = output.total_size();
    dx.reshape(output.shape);
    for (size_t i = 0; i < sz; ++i) {
        float s = output.data[i];
        dx.data[i] = dout.data[i] * s * (1.0f - s);
    }
}

void tanh_forward(const Tensor& input, Tensor& output) {
    size_t sz = input.total_size();
    output.reshape(input.shape);
    for (size_t i = 0; i < sz; ++i) {
        output.data[i] = std::tanh(input.data[i]);
    }
}

void tanh_backward(const Tensor& dout, const Tensor& output, Tensor& dx) {
    size_t sz = output.total_size();
    dx.reshape(output.shape);
    for (size_t i = 0; i < sz; ++i) {
        float th = output.data[i];
        dx.data[i] = dout.data[i] * (1.0f - th * th);
    }
}

void elementwise_add(const Tensor& a, const Tensor& b, Tensor& out) {
    size_t sz = a.total_size();
    out.reshape(a.shape);
    for (size_t i = 0; i < sz; ++i) out.data[i] = a.data[i] + b.data[i];
}

void elementwise_sub(const Tensor& a, const Tensor& b, Tensor& out) {
    size_t sz = a.total_size();
    out.reshape(a.shape);
    for (size_t i = 0; i < sz; ++i) out.data[i] = a.data[i] - b.data[i];
}

void elementwise_mul(const Tensor& a, const Tensor& b, Tensor& out) {
    size_t sz = a.total_size();
    out.reshape(a.shape);
    for (size_t i = 0; i < sz; ++i) out.data[i] = a.data[i] * b.data[i];
}

} // namespace ns
