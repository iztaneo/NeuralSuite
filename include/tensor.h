// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file tensor.h
 * @brief Multi-dimensional Tensor & Linear Algebra Engine strictly following the Google C++ Style Guide.
 * @see https://google.github.io/styleguide/cppguide.html
 */

#ifndef NEURAL_SUITE_INCLUDE_TENSOR_H_
#define NEURAL_SUITE_INCLUDE_TENSOR_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuralsuite {

/**
 * @brief Pi, definido aqui y no tomado de `M_PI`.
 *
 * `M_PI` es una extension POSIX, no parte de C++ estandar. La glibc y la
 * libc++ lo definen en `<cmath>` y por eso compilaba en Linux y en macOS; MSVC
 * solo lo define si se pide antes `_USE_MATH_DEFINES`, de modo que la
 * compilacion de Windows llevaba dias rota sin que se notara en local.
 *
 * Se define aqui en vez de activar esa macro: una constante `constexpr` no
 * depende del orden de los `#include` ni de que nadie se acuerde de ponerla.
 */
constexpr double kPi = 3.14159265358979323846;

/**
 * @class Tensor
 * @brief Multi-dimensional array storing floating point numbers in contiguous memory.
 * @details Conforms to Google C++ Style Guide: RAII memory management, explicit constructors,
 * const-correctness, y almacenamiento compartido que permite vistas sin copia.
 */
class Tensor {
 public:
  Tensor();
  explicit Tensor(const std::vector<int>& dims);
  Tensor(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  ~Tensor() = default;

  Tensor& operator=(const Tensor& other);
  Tensor& operator=(Tensor&& other) noexcept;

  // Accessors
  [[nodiscard]] size_t TotalSize() const;
  [[nodiscard]] const std::vector<int>& Shape() const { return shape_; }
  [[nodiscard]] float* Data() { return storage_ ? storage_->data() + offset_ : nullptr; }
  [[nodiscard]] const float* Data() const {
    return storage_ ? storage_->data() + offset_ : nullptr;
  }

  /**
   * @brief Devuelve una vista con otra forma que comparte la misma memoria.
   *
   * No copia nada: escribir en la vista modifica el tensor original. El número
   * de elementos debe coincidir. Sirve para las reinterpretaciones frecuentes
   * entre `[B, T, C]` y `[B*T, C]`, que antes obligaban a reservar un tensor
   * nuevo y hacer un `memcpy` completo en cada capa densa.
   *
   * La memoria vive mientras exista cualquiera de los dos tensores. Copiar la
   * vista con el constructor de copia o `operator=` sí produce un tensor
   * independiente, de modo que el código que guarda instantáneas
   * (`last_input_ = input`) conserva su comportamiento.
   */
  [[nodiscard]] Tensor View(const std::vector<int>& new_shape) const;

  /** @brief Indica si esta instancia comparte memoria con otra. */
  [[nodiscard]] bool SharesStorageWith(const Tensor& other) const {
    return storage_ && storage_ == other.storage_;
  }

  /**
   * @brief Reinterpreta las dimensiones conservando la memoria y los datos.
   *
   * El número total de elementos debe coincidir con el actual; en caso
   * contrario lanza std::invalid_argument. Ésta es la semántica habitual de
   * reshape: misma memoria, distinta lectura de sus ejes. Para cambiar el
   * tamaño del almacenamiento, usar Resize().
   */
  void Reshape(const std::vector<int>& new_shape);

  /**
   * @brief Ajusta la forma reasignando memoria cuando el tamaño cambia.
   *
   * Si el número de elementos difiere del actual, el contenido anterior se
   * descarta y el nuevo buffer queda a cero. Es lo que necesitan los tensores
   * de salida antes de escribirlos.
   */
  void Resize(const std::vector<int>& new_shape);

  void Fill(float val);
  void Zeros();
  void Ones();
  void RandomNormal(float mean = 0.0f, float stddev = 0.02f);
  void RandomUniform(float min_val = -0.1f, float max_val = 0.1f);
  void XavierInit(int fan_in, int fan_out);

  // Element access operators. La comprobación de límites solo se compila en
  // builds de depuración: en Release el acceso debe seguir siendo un simple
  // desplazamiento sobre el puntero.
#ifdef NDEBUG
  inline float& operator[](size_t idx) { return storage_->data()[offset_ + idx]; }
  inline const float& operator[](size_t idx) const { return storage_->data()[offset_ + idx]; }
#else
  inline float& operator[](size_t idx) {
    CheckIndex(idx);
    return storage_->data()[offset_ + idx];
  }
  inline const float& operator[](size_t idx) const {
    CheckIndex(idx);
    return storage_->data()[offset_ + idx];
  }
#endif

  void PrintSummary(const std::string& name = "") const;

 private:
  // Almacenamiento contiguo compartido entre un tensor y sus vistas. Se usa
  // std::vector como bloque de memoria: da liberación automática y evita el
  // new[]/delete[] manual que obligaba a escribir a mano las cuatro
  // operaciones de copia y movimiento.
  using Storage = std::vector<float>;
#ifndef NDEBUG
  void CheckIndex(size_t idx) const {
    if (idx >= TotalSize()) {
      throw std::out_of_range("Tensor: indice " + std::to_string(idx) +
                              " fuera de rango para un tensor de " +
                              std::to_string(TotalSize()) + " elementos.");
    }
  }
#endif

  std::shared_ptr<Storage> storage_;
  size_t offset_ = 0;
  std::vector<int> shape_;
};

/**
 * @brief Fija la semilla del generador que usan RandomNormal, RandomUniform y
 *        XavierInit.
 *
 * Sin esto la unica forma de reproducir una inicializacion era confiar en que
 * nadie hubiera consumido numeros antes: el generador es global, asi que
 * construir una capa de mas desplaza todo lo que venga despues. Eso convertia
 * en fragiles las comparaciones entre ejecuciones.
 *
 * El generador sigue siendo compartido: no es seguro usarlo desde varios hilos
 * a la vez, y aislarlo por hilo queda pendiente.
 */
void ManualSeed(uint32_t seed);

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
 * @brief Une varios tensores a lo largo de un eje.
 *
 * Todas las entradas deben coincidir en forma salvo en el eje `eje`, que es el
 * que se suma. Un eje negativo cuenta desde el final, como es costumbre.
 *
 * Es la operacion que necesitan los saltos de una U-Net —el bloque de subida
 * concatena su entrada con la salida guardada del bloque de bajada del mismo
 * nivel— y tambien la forma natural de juntar las cabezas de una atencion.
 */
Tensor Concat(const std::vector<const Tensor*>& entradas, int eje);

/** @brief Atajo para el caso de dos tensores, que es el habitual. */
Tensor Concat(const Tensor& a, const Tensor& b, int eje);

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
 * @brief SiLU (Swish): x * sigmoid(x).
 *
 * Es la activacion de los transformers modernos —LLaMA, Mistral, Gemma— y de
 * las U-Net de difusion, asi que sirve a los dos lados del framework.
 *
 * Frente a ReLU no se corta en cero: para x negativa deja pasar un poco, y esa
 * cola es lo que evita que una neurona se apague del todo y deje de recibir
 * gradiente. Frente a GELU es mas barata —una exponencial en vez de una
 * tangente hiperbolica sobre un polinomio— y muy parecida en forma.
 *
 * El backward necesita la ENTRADA, no la salida, porque su derivada
 * `s * (1 + x * (1 - s))` con `s = sigmoid(x)` no se puede recuperar de `y`
 * sola: SiLU no es inyectiva. Tiene un minimo cerca de x = -1.278, asi que hay
 * dos valores de x que dan la misma y. Es la diferencia con `SigmoidBackward`,
 * que si toma la salida, y confundirlas da un gradiente equivocado justo en la
 * zona negativa que es la razon de usar SiLU.
 */
void SiluForward(const Tensor& input, Tensor& output);
void SiluBackward(const Tensor& dout, const Tensor& input, Tensor& dx);

/**
 * @brief RoPE: codifica la posicion rotando pares de canales.
 *
 * `x` es `[lote, pasos, n_head * hd]`. Cada cabeza se rota por su cuenta, y
 * dentro de ella los canales se toman **de dos en dos y adyacentes** —(0,1),
 * (2,3), ...—, que es la convencion del articulo original. LLaMA usa otra
 * —parte la cabeza por la mitad y empareja `i` con `i + hd/2`— y **las dos no
 * son intercambiables**: producen rotaciones distintas sobre los mismos
 * numeros. Mezclarlas es el error silencioso de esta operacion, porque el
 * resultado sigue siendo una rotacion valida y nada falla.
 *
 * El angulo del par `i` en la posicion `p` es `p / base^(2i/hd)`. Pares
 * distintos giran a velocidades distintas, y de ahi sale que el producto
 * `q·k` acabe dependiendo solo de la **diferencia** de posiciones: la posicion
 * entra como una rotacion, y el producto escalar entre dos vectores rotados
 * depende del angulo relativo. Por eso deslizar una ventana no invalida nada,
 * que es lo que a los embeddings aprendidos les cuesta caro.
 *
 * `pos_inicial` es la posicion absoluta del primer paso. Con KV-Cache hay que
 * pasar la posicion REAL en la secuencia, no el indice dentro de la cache: es
 * el error clasico de esta combinacion.
 *
 * La rotacion es ortogonal, asi que su gradiente es la rotacion inversa. Eso
 * hace el backward barato y da una propiedad comprobable: rotar y desrotar
 * devuelve el punto de partida.
 */
void RopeForward(const Tensor& input, Tensor& output, int n_head, int pos_inicial = 0,
                 float base = 10000.0f);

/** @brief Gradiente de `RopeForward`: la rotacion por el angulo opuesto. */
void RopeBackward(const Tensor& dout, Tensor& dx, int n_head, int pos_inicial = 0,
                  float base = 10000.0f);

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
