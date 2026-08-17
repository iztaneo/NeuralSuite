# 📖 Guía de Programación C++17 y Arquitectura de Sistemas en NeuralSuite

Esta guía detalla las **decisiones de ingeniería de software, patrones de diseño C++17, gestión de memoria RAII y optimizaciones de rendimiento** utilizadas en NeuralSuite.

---

## 1. Patrón Orientado a Objetos (Grafo de Capas)

NeuralSuite implementa la abstracción de capas jerárquicas en C++17 mediante polimorfismo puro:

```cpp
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_gradients() { return {}; }
};
```

---

## 2. Gestión de Memoria Dinámica (RAII)

Para garantizar cero fugas de memoria (*memory leaks*) sin depender de recogedores de basura, el tipo `Tensor` utiliza el principio **RAII (Resource Acquisition Is Initialization)**:

```cpp
class Tensor {
public:
    float* data = nullptr;
    std::vector<int> shape;

    Tensor(const std::vector<int>& dims) : shape(dims) {
        size_t size = total_size();
        data = new float[size]();
    }

    ~Tensor() {
        delete[] data;
    }

    // Semántica de movimiento (Move semantics) C++11/C++17 para eficiencia
    Tensor(Tensor&& other) noexcept {
        data = other.data;
        shape = std::move(other.shape);
        other.data = nullptr;
    }
};
```

---

## 3. Paralelización en CPU con OpenMP y Vectorización SIMD

### Directivas OpenMP en GEMM (Multiplicación de Matrices)
```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

### Banderas de Compilación Requeridas
- **Linux (`g++`)**: `-O3 -std=c++17 -fopenmp -march=native -ftree-vectorize`
- **macOS (`clang++`)**: `-O3 -std=c++17 -Xpreprocessor -fopenmp -march=native`
- **Windows (`cl.exe` / `CMake`)**: `/O2 /std:c++17 /openmp`
