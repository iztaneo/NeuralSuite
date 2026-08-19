// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.
//
// Compara el reparto propio (std::thread) contra OpenMP sobre el MISMO bucle y
// los mismos datos. Sirve para responder una pregunta concreta: si renunciar a
// OpenMP cuesta rendimiento, dado que sus implementaciones llevan anos
// afinandose y la de aqui es casera.
//
// No forma parte del build normal, porque compilarlo exige OpenMP y el
// proyecto ya no lo usa. Se compila a mano cuando se quiera repetir la medida:
//
//   Linux:  g++ -std=c++17 -Iinclude -O2 -DNDEBUG -fopenmp \
//               benchmarks/compare_openmp.cpp src/*.cpp -o compare_openmp
//   macOS:  g++ -std=c++17 -Iinclude -O2 -DNDEBUG -Xpreprocessor -fopenmp \
//               -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp \
//               benchmarks/compare_openmp.cpp src/*.cpp -o compare_openmp
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <omp.h>
#include "parallel.h"
#include "tensor.h"
using namespace neuralsuite;

static void MatMulOmp(const Tensor& A, const Tensor& B, Tensor& C){
  const int M=A.Shape()[0], K=A.Shape()[1], N=B.Shape()[1];
  C.Resize({M,N}); C.Zeros();
  #pragma omp parallel for schedule(static)
  for(int i=0;i<M;++i)
    for(int k=0;k<K;++k){
      const float a=A[i*K+k];
      const float* br=&B[k*N];
      float* cr=&C[i*N];
      for(int j=0;j<N;++j) cr[j]+=a*br[j];
    }
}

template<typename Fn> static double med(Fn&& f,int reps){
  std::vector<double> v; f();
  for(int i=0;i<reps;++i){
    auto t0=std::chrono::steady_clock::now(); f();
    auto t1=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
  }
  std::sort(v.begin(),v.end()); return v[v.size()/2];
}

static void run(int M,int K,int N){
  Tensor A({M,K}),B({K,N}),C1,C2;
  A.RandomNormal(0,1); B.RandomNormal(0,1);
  MatMul(A,B,C1);                                  // crea el pool al maximo
  const double mine=med([&]{ MatMul(A,B,C1); },15);
  const double omp =med([&]{ MatMulOmp(A,B,C2); },15);
  double d=0; for(size_t i=0;i<C1.TotalSize();++i) d=std::max(d,(double)std::abs(C1[i]-C2[i]));
  const double gf=2.0*M*K*N/1e6;
  printf("  %4dx%4dx%-4d  propio %7.3f ms (%6.1f GF)   OpenMP %7.3f ms (%6.1f GF)   ratio %.2fx  dif=%.1e\n",
         M,K,N, mine, gf/mine, omp, gf/omp, omp/mine, d);
}
int main(){
  printf("hardware_concurrency = %u\n", std::thread::hardware_concurrency());
  printf("parallel::ThreadCount() = %d\n", parallel::ThreadCount());
  #pragma omp parallel
  {
    #pragma omp single
    printf("omp_get_num_threads = %d\n", omp_get_num_threads());
  }
  printf("\n");
  printf("Reparto propio (std::thread) frente a OpenMP, mismo bucle:\n");
  printf("  ratio > 1 significa que el propio es mas rapido\n\n");
  run(512,128,384);
  run(512,128,512);
  run(512,512,512);
  run(1024,512,512);
  run(2048,512,512);
  return 0;
}
