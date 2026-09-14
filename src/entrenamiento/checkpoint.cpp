// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

// Implementacion de entrenamiento/checkpoint.h.

#include "entrenamiento/checkpoint.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace neuralsuite {
namespace entrenamiento {

void RegistroSellado::MarcarDados(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--", 2) == 0) dados_.insert(argv[i] + 2);
  }
}

void RegistroSellado::Entero(const std::string& nombre, int* variable) {
  entradas_[nombre] = {[variable] { return std::to_string(*variable); },
                       [variable](const std::string& v) { *variable = std::stoi(v); }};
}

void RegistroSellado::Real(const std::string& nombre, float* variable) {
  entradas_[nombre] = {[variable] { return TextoReal(*variable); },
                       [variable](const std::string& v) { *variable = std::stof(v); }};
}

void RegistroSellado::Texto(const std::string& nombre, std::string* variable) {
  entradas_[nombre] = {[variable] { return *variable; },
                       [variable](const std::string& v) { *variable = v; }};
}

bool RegistroSellado::Adoptar(const Metadatos& sellado, std::string* adoptados,
                              std::string* error) {
  adoptados->clear();
  for (const auto& [nombre, entrada] : entradas_) {
    const auto it = sellado.find(nombre);
    if (it == sellado.end()) continue;
    const std::string actual = entrada.leer();
    if (Dado(nombre)) {
      if (actual != it->second) {
        *error = "El checkpoint se entreno con " + nombre + "=" + it->second + " y se pidio " +
                 actual;
        return false;
      }
    } else if (actual != it->second) {
      entrada.fijar(it->second);
      *adoptados += " " + nombre + "=" + it->second;
    }
  }
  return true;
}

Metadatos RegistroSellado::Valores() const {
  Metadatos m;
  for (const auto& [nombre, entrada] : entradas_) m[nombre] = entrada.leer();
  return m;
}

std::string TextoReal(double valor) {
  char b[40];
  std::snprintf(b, sizeof(b), "%.9g", valor);
  return std::string(b);
}

float TasaAprendizaje(int it, int total, int calentamiento, float lr, float lr_min) {
  if (calentamiento > 0 && it <= calentamiento) {
    return lr * static_cast<float>(it) / static_cast<float>(calentamiento);
  }
  if (total > calentamiento) {
    const float avance =
        static_cast<float>(it - calentamiento) / static_cast<float>(total - calentamiento);
    return lr_min + 0.5f * (lr - lr_min) * (1.0f + std::cos(3.14159265f * avance));
  }
  return lr;
}

uint32_t SemillaIteracion(int semilla, int it) {
  return (static_cast<uint32_t>(semilla) * 0x9E3779B1u) ^
         (static_cast<uint32_t>(it) * 0x85EBCA6Bu);
}

std::string RutaArchivada(const std::string& base, int it) {
  char sufijo[32];
  std::snprintf(sufijo, sizeof(sufijo), "_it%06d", it);
  const size_t punto = base.rfind('.');
  const size_t barra = base.find_last_of("/\\");
  // Un punto dentro de un directorio (`./release/x`) no es una extension.
  if (punto == std::string::npos || (barra != std::string::npos && punto < barra)) {
    return base + sufijo;
  }
  return base.substr(0, punto) + sufijo + base.substr(punto);
}

bool GuardarCheckpoint(const std::vector<Parte>& partes, int iteracion, Metadatos sello,
                       std::string* error) {
  const uint64_t id =
      (static_cast<uint64_t>(iteracion) << 40) ^
      static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
  sello["checkpoint_id"] = std::to_string(id);
  sello["iteracion"] = std::to_string(iteracion);

  std::vector<std::string> temporales;
  bool ok = true;
  for (const Parte& p : partes) {
    temporales.push_back(p.ruta + ".tmp");
    if (!p.escribir(temporales.back(), sello)) {
      *error = "no se pudo escribir " + temporales.back();
      ok = false;
      break;
    }
  }
  if (!ok) {
    // El checkpoint anterior sigue intacto: no se ha movido nada.
    for (const std::string& t : temporales) std::remove(t.c_str());
    return false;
  }
  for (size_t k = 0; k < partes.size(); ++k) {
    if (std::rename(temporales[k].c_str(), partes[k].ruta.c_str()) != 0) {
      *error = "no se pudo mover " + temporales[k] + " a su sitio";
      return false;
    }
  }
  return true;
}

bool ComprobarMismoCheckpoint(const std::vector<Metadatos>& metadatos,
                              const std::vector<std::string>& nombres, std::string* error) {
  if (metadatos.empty()) return true;
  const auto leer = [](const Metadatos& m) {
    const auto it = m.find("checkpoint_id");
    return it == m.end() ? std::string() : it->second;
  };
  const std::string id = leer(metadatos[0]);
  bool igual = !id.empty();
  for (const Metadatos& m : metadatos) igual = igual && leer(m) == id;
  if (igual) return true;
  *error = "Los archivos no son del mismo checkpoint (";
  for (size_t i = 0; i < metadatos.size(); ++i) {
    *error += (i ? ", " : "") + (i < nombres.size() ? nombres[i] : "?") + " '" + leer(metadatos[i]) + "'";
  }
  *error += "). Probablemente un corte durante el guardado; usa un checkpoint anterior.";
  return false;
}

namespace {

std::vector<nsf::NamedTensor> TensoresAdam(AdamW& opt) {
  std::vector<nsf::NamedTensor> est;
  const auto ms = opt.EstadoM();
  const auto vs = opt.EstadoV();
  for (size_t i = 0; i < ms.size(); ++i) {
    est.push_back({"m." + std::to_string(i), ms[i]});
    est.push_back({"v." + std::to_string(i), vs[i]});
  }
  return est;
}

}  // namespace

nsf::Result GuardarEstadoAdam(const std::string& ruta, AdamW& opt, Metadatos meta) {
  meta["arch"] = "adamw";
  meta["pasos_opt"] = std::to_string(opt.PasosDados());
  return nsf::Save(ruta, TensoresAdam(opt), meta);
}

nsf::Result CargarEstadoAdam(const std::string& ruta, AdamW& opt, Metadatos* meta) {
  const auto r = nsf::Load(ruta, TensoresAdam(opt), {{"arch", "adamw"}}, meta);
  if (r.ok) opt.FijarPasosDados(std::stoi((*meta)["pasos_opt"]));
  return r;
}

}  // namespace entrenamiento
}  // namespace neuralsuite
