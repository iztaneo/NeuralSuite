// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.
//
// Responde si conviene que `Transpose` sea una vista en lugar de una copia, que
// es lo que permitirian los strides en `Tensor`.
//
// Toda transposicion del repositorio alimenta directamente un `MatMul`, asi que
// la pregunta se reduce a: compensa leer el operando transpuesto en vez de
// copiarlo primero. La respuesta medida es que no, y por un margen grande: al
// leer transpuesto cada elemento del resultado pasa a ser un producto escalar,
// con una dependencia en el bucle interno que atasca el pipeline.
//
// Requiere MatMulNT y MatMulTN, que no estan en la biblioteca precisamente por
// este resultado. El archivo se conserva como registro de la medicion; para
// repetirla hay que reintroducirlas.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#include "tensor.h"
using namespace neuralsuite;
template<typename F> static double med(F&&f,int r){
  std::vector<double> v; f();
  for(int i=0;i<r;++i){auto a=std::chrono::steady_clock::now();f();
    auto b=std::chrono::steady_clock::now();
    v.push_back(std::chrono::duration<double,std::milli>(b-a).count());}
  std::sort(v.begin(),v.end()); return v[v.size()/2];
}
int main(){
  printf("Transponer+MatMul frente a las variantes fusionadas (GFLOP/s):\n\n");
  printf("  %-18s %12s %12s\n","forma","transp+MM","fusionada");
  for(auto d:{std::array<int,3>{512,128,384},{512,512,512},{1024,512,512}}){
    const int M=d[0],K=d[1],N=d[2];
    const double gf=2.0*M*K*N/1e6;
    { // NT: C = A * B^T,  A[M,K], B[N,K]
      Tensor A({M,K}),B({N,K}),C; A.RandomNormal(0,1); B.RandomNormal(0,1);
      const double old=med([&]{ Tensor bt=Transpose(B); MatMul(A,bt,C); },10);
      const double neu=med([&]{ MatMulNT(A,B,C); },10);
      char sh[24]; snprintf(sh,sizeof(sh),"NT %dx%dx%d",M,K,N);
      printf("  %-18s %12.1f %12.1f\n",sh,gf/old,gf/neu);
    }
    { // TN: C = A^T * B,  A[K,M], B[K,N]
      Tensor A({K,M}),B({K,N}),C; A.RandomNormal(0,1); B.RandomNormal(0,1);
      const double old=med([&]{ Tensor at=Transpose(A); MatMul(at,B,C); },10);
      const double neu=med([&]{ MatMulTN(A,B,C); },10);
      char sh[24]; snprintf(sh,sizeof(sh),"TN %dx%dx%d",M,K,N);
      printf("  %-18s %12.1f %12.1f\n",sh,gf/old,gf/neu);
    }
  }
  return 0;
}
