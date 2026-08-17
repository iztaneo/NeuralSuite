// Copyright 2026 NeuralSuite Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * @file artifacts.h
 * @brief Convención de rutas para artefactos generados (pesos y vocabularios).
 *
 * Todo modelo entrenado se escribe bajo el directorio `release/`, nunca en la
 * raíz del repositorio. El directorio está excluido del control de versiones:
 * los checkpoints que se quieran publicar se adjuntan a un GitHub Release, no
 * se commitean al historial.
 */

#ifndef NEURAL_SUITE_INCLUDE_ARTIFACTS_H_
#define NEURAL_SUITE_INCLUDE_ARTIFACTS_H_

#include <filesystem>
#include <string>

namespace neuralsuite {

/** @brief Directorio donde viven los artefactos generados por entrenamiento. */
inline constexpr const char* kReleaseDir = "release";

/**
 * @brief Devuelve `release/<filename>`, creando el directorio si no existe.
 *
 * Si el directorio no se puede crear se devuelve `filename` sin prefijo, de
 * modo que el llamador escriba en el directorio actual en lugar de fallar en
 * silencio contra una ruta inexistente.
 */
inline std::string ReleasePath(const std::string& filename) {
  std::error_code ec;
  std::filesystem::create_directories(kReleaseDir, ec);
  if (ec) return filename;
  return (std::filesystem::path(kReleaseDir) / filename).string();
}

}  // namespace neuralsuite

#endif  // NEURAL_SUITE_INCLUDE_ARTIFACTS_H_
