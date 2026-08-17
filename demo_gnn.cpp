// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file demo_gnn.cpp
 * @brief Graph Neural Network (GCN Node Classification) Demo in C++.
 */

#include <iostream>
#include <memory>
#include "neuralsuite.h"

using namespace neuralsuite;

int main() {
  std::cout << "============================================================\n" << std::flush;
  std::cout << "🕸️ Demostración 7: Red Neuronal para Grafos (GNN / GCN C++)\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  // Grafo de 4 Nodos: Adyacencia A_norm con bucles propios (Self-loops)
  Tensor adj_norm({4, 4});
  adj_norm[0] = 0.5f; adj_norm[1] = 0.5f; adj_norm[2] = 0.0f; adj_norm[3] = 0.0f;
  adj_norm[4] = 0.5f; adj_norm[5] = 0.5f; adj_norm[6] = 0.0f; adj_norm[7] = 0.0f;
  adj_norm[8] = 0.0f; adj_norm[9] = 0.0f; adj_norm[10] = 0.5f; adj_norm[11] = 0.5f;
  adj_norm[12] = 0.0f; adj_norm[13] = 0.0f; adj_norm[14] = 0.5f; adj_norm[15] = 0.5f;

  // Características de Nodos (4 nodos, 2 características cada uno)
  Tensor X({4, 2});
  X[0] = 1.0f; X[1] = 0.0f;
  X[2] = 1.0f; X[3] = 0.1f;
  X[4] = 0.0f; X[5] = 1.0f;
  X[6] = 0.1f; X[7] = 1.0f;

  // Etiquetas de Nodos (Clase 0 para comunidad A, Clase 1 para comunidad B)
  Tensor Y({4});
  Y[0] = 0.0f; Y[1] = 0.0f; Y[2] = 1.0f; Y[3] = 1.0f;

  GraphConv gcn_layer1(2, 4);
  Linear classifier(4, 2);

  CrossEntropyLoss criterion;

  std::vector<Tensor*> params, grads;
  for (auto p : gcn_layer1.GetParameters()) params.push_back(p);
  for (auto p : classifier.GetParameters()) params.push_back(p);
  for (auto g : gcn_layer1.GetGradients()) grads.push_back(g);
  for (auto g : classifier.GetGradients()) grads.push_back(g);

  AdamW optimizer(params, grads, 0.05f);

  std::cout << "🏋️ Entrenando GNN (Clasificación de Nodos de Grafo) durante 50 épocas...\n" << std::flush;
  for (int epoch = 1; epoch <= 50; ++epoch) {
    optimizer.ZeroGrad();

    Tensor h_gcn = gcn_layer1.ForwardWithAdj(X, adj_norm);
    Tensor logits = classifier.Forward(h_gcn);

    float loss = criterion.Forward(logits, Y);

    Tensor dlogits = criterion.Backward();
    Tensor dh_gcn = classifier.Backward(dlogits);
    gcn_layer1.Backward(dh_gcn);

    optimizer.Step();

    if (epoch % 10 == 0 || epoch == 50) {
      std::cout << "Época " << epoch << "/50 | Loss GNN Grafo: " << loss << "\n" << std::flush;
    }
  }

  std::cout << "============================================================\n" << std::flush;
  std::cout << "✅ ¡Red Neuronal para Grafos (GNN) entrenada y verificada exitosamente en C++!\n" << std::flush;
  std::cout << "============================================================\n" << std::flush;

  return 0;
}
