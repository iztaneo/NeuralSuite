// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file parallel.h
 * @brief Reparto de bucles entre hilos, con la biblioteca estandar.
 *
 * El paralelismo dependia de OpenMP, que no esta disponible en todas las
 * plataformas que el proyecto soporta: con AppleClang los `pragma` se ignoran y
 * todo corre en un solo hilo. Medido sobre un Apple M5 de cuatro nucleos de
 * rendimiento, eso dejaba `MatMul` —que es el 80% del tiempo de un paso de
 * entrenamiento— usando una cuarta parte de la maquina.
 *
 * Aqui el reparto se hace con `std::thread`, que existe en cualquier
 * compilador de C++17, de modo que el comportamiento es el mismo en las tres
 * plataformas y no hace falta ninguna dependencia externa.
 *
 * Cabe preguntarse si renunciar a OpenMP cuesta rendimiento, ya que sus
 * implementaciones llevan anos afinandose. Se midio en las dos plataformas,
 * repartiendo el mismo bucle de ambas formas (benchmarks/compare_openmp.cpp):
 *
 *   Apple M5, 10 hilos     propio y OpenMP entre 0.88x y 1.14x, media ~1.02x
 *   Linux CI, 4 vCPU       serie 3.1 GF; propio 1.76x, OpenMP 2.00x
 *
 * En macOS estan a la par. En Linux OpenMP saca alrededor de un 12%, no el 2x
 * que sugerian las primeras cifras: aquellas se tomaron sin medir el caso de un
 * solo hilo, y comparar dos aceleraciones sin su referencia no dice nada. La
 * leccion vale mas que el numero.
 *
 * Ese 12% no compensa mantener dos rutas: habria que probar cada bucle por
 * duplicado, el OpenMP 2.0 de MSVC obliga a indices con signo y prohibe `omp
 * simd`, y dos repartos distintos pueden divergir en cuanto alguna operacion
 * lleve una reduccion. Sobre todo, OpenMP falla en silencio cuando no esta: el
 * `pragma` se ignora y todo corre en un hilo sin ningun aviso, que es
 * exactamente como este proyecto llego a tener macOS sin paralelizar.
 */

#ifndef NEURAL_SUITE_INCLUDE_PARALLEL_H_
#define NEURAL_SUITE_INCLUDE_PARALLEL_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace neuralsuite {
namespace parallel {

/**
 * @brief Numero de hilos que se usan para repartir los bucles.
 *
 * Por defecto, el numero de nucleos que informa el sistema. Fijarlo a 1
 * desactiva el paralelismo, lo que resulta util para medir o para depurar.
 */
inline int& ThreadCount() {
  static int count = []() {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
  }();
  return count;
}

namespace detail {

/** @brief Marca si el hilo actual ya esta ejecutando trabajo repartido. */
inline bool& InsideParallelRegion() {
  static thread_local bool inside = false;
  return inside;
}

/**
 * @class Pool
 * @brief Hilos persistentes que esperan trabajo.
 *
 * Crear hilos en cada llamada costaria mas que el propio calculo en las
 * matrices pequenas, y un paso de entrenamiento hace decenas de
 * multiplicaciones. Los hilos se crean una vez y se reutilizan.
 */
class Pool {
 public:
  static Pool& Instance() {
    static Pool pool;
    return pool;
  }

  /** @brief Cuantos indices de trabajo admite: los hilos propios mas el llamante. */
  [[nodiscard]] int Capacity() const { return static_cast<int>(threads_.size()) + 1; }

  /** @brief Ejecuta `fn(worker_index)` en N indices y espera a que terminen. */
  void Run(int workers, const std::function<void(int)>& fn) {
    // El pool se crea una sola vez, con el numero de hilos que hubiera entonces.
    // Si despues alguien sube ThreadCount(), pedir mas indices de los que hay
    // dejaria porciones del bucle sin ejecutar y el resultado saldria mal sin
    // ningun aviso. Se recorta aqui para que eso no pueda ocurrir.
    workers = std::min(workers, Capacity());
    {
      std::unique_lock<std::mutex> lock(mutex_);
      task_ = &fn;
      pending_ = workers - 1;   // el hilo llamante hace la primera porcion
      remaining_ = pending_;
      ++generation_;
      work_ready_.notify_all();
    }

    // El llamante trabaja tambien: asi con un solo hilo no hay ningun coste de
    // sincronizacion, y con varios se aprovecha su nucleo.
    //
    // Y mientras lo hace tiene que contar como dentro de la region paralela.
    // WorkerLoop marca sus hilos, pero el llamante no se marcaba a si mismo, de
    // modo que si su porcion volvia a pedir reparto -por ejemplo una
    // convolucion que reparte por lote y llama a MatMul, que reparte por
    // filas- se reentraba en el pool con trabajo aun en vuelo: se pisaba la
    // tarea, se adelantaba la generacion y la espera de fuera no se despertaba
    // nunca. El bloqueo era real y estuvo latente hasta que algo anido de
    // verdad.
    {
      class Marca {
       public:
        Marca() : previo_(InsideParallelRegion()) { InsideParallelRegion() = true; }
        ~Marca() { InsideParallelRegion() = previo_; }
       private:
        bool previo_;
      } marca;
      fn(0);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [this] { return remaining_ == 0; });
    task_ = nullptr;
  }

  ~Pool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
      work_ready_.notify_all();
    }
    for (std::thread& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

 private:
  Pool() {
    const int n = std::max(1, ThreadCount()) - 1;
    for (int i = 0; i < n; ++i) {
      threads_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }
  }

  void WorkerLoop(int index) {
    detail::InsideParallelRegion() = true;
    unsigned long long seen = 0;
    while (true) {
      const std::function<void(int)>* task = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [&] { return stop_ || (generation_ != seen && index <= pending_); });
        if (stop_) return;
        seen = generation_;
        task = task_;
      }
      if (task) (*task)(index);
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--remaining_ == 0) work_done_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  const std::function<void(int)>* task_ = nullptr;
  unsigned long long generation_ = 0;
  int pending_ = 0;
  int remaining_ = 0;
  bool stop_ = false;
};

}  // namespace detail

/**
 * @brief Reparte `[0, count)` entre los hilos disponibles.
 *
 * `fn(begin, end)` recibe un rango contiguo y no solapado, de modo que las
 * porciones no necesitan sincronizarse entre si. Si cada hilo escribe en zonas
 * distintas del resultado —como las filas de una matriz—, el resultado es
 * identico bit a bit al de un solo hilo: no hay reduccion que cambie el orden
 * de las sumas en punto flotante.
 *
 * El reparto es dinamico: los rangos no estan asignados de antemano a un hilo
 * concreto, sino que cada uno toma el siguiente bloque libre. Eso no afecta al
 * resultado —los rangos siguen siendo disjuntos— pero si a que hilo le toca
 * cada parte, de modo que no conviene depender de esa correspondencia.
 *
 * `min_per_thread` evita repartir trabajo tan pequeno que la sincronizacion
 * cueste mas que el calculo.
 */
inline void ParallelFor(int count, int min_per_thread,
                        const std::function<void(int begin, int end)>& fn) {
  if (count <= 0) return;

  int workers = std::max(1, ThreadCount());
  workers = std::min(workers, std::max(1, count / std::max(1, min_per_thread)));

  // Anidar reparto dentro de reparto agotaria los hilos y podria bloquear: si
  // ya estamos dentro de una region paralela, se ejecuta en serie.
  if (workers <= 1 || detail::InsideParallelRegion()) {
    fn(0, count);
    return;
  }

  detail::Pool& pool = detail::Pool::Instance();
  workers = std::min(workers, pool.Capacity());
  if (workers <= 1) {
    fn(0, count);
    return;
  }

  // Reparto dinamico: el trabajo se divide en mas bloques que hilos y cada uno
  // toma el siguiente en cuanto se desocupa.
  //
  // Con trozos iguales —uno por hilo— el tiempo total lo marca el hilo mas
  // lento, y los nucleos rara vez son iguales: en un Apple M5 conviven nucleos
  // de rendimiento y de eficiencia, y en un servidor dos hilos pueden compartir
  // el mismo nucleo fisico. Medido sobre esta maquina, pasar de trozos iguales
  // a bloques que se reparten sobre la marcha sube la aceleracion de 4.6x a
  // 5.2x, que es justo el margen que nos separaba de OpenMP.
  //
  // El tamano de bloque equilibra dos cosas: bloques pequenos reparten mejor,
  // pero cada uno cuesta una operacion atomica y arruina la localidad si son
  // demasiado finos.
  const int block = std::max(min_per_thread, count / (workers * 4));
  std::atomic<int> next_block{0};

  pool.Run(workers, [&fn, &next_block, block, count](int) {
    for (;;) {
      const int begin = next_block.fetch_add(block, std::memory_order_relaxed);
      if (begin >= count) return;
      fn(begin, std::min(count, begin + block));
    }
  });
}

}  // namespace parallel
}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_PARALLEL_H_
