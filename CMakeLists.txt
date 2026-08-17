cmake_minimum_required(VERSION 3.14)
project(NeuralSuite VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sin un tipo de build explícito, CMake no define NDEBUG y la comprobación de
# límites de Tensor::operator[] queda activa dentro de un binario optimizado.
# Release es el comportamiento esperado por defecto; para obtener el build con
# comprobaciones: cmake -B build -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# OpenMP acelera algunos bucles pero no es imprescindible: sin el, los pragmas
# se ignoran y todo se ejecuta secuencialmente. Exigirlo con REQUIRED rompia la
# configuracion en macOS, porque AppleClang no lo trae, de modo que el build
# documentado en el README no funcionaba en una de las plataformas que anuncia.
option(NEURALSUITE_ENABLE_OPENMP "Usar OpenMP si esta disponible" ON)

# -march=native compila para la CPU de la maquina actual: va bien para medir en
# local, pero produce binarios que pueden no arrancar en otro procesador.
# -ffast-math relaja las garantias IEEE-754 (NaN, infinitos, asociatividad), lo
# que estorba justo cuando se verifican gradientes. Ambas quedan desactivadas
# por defecto y se piden explicitamente para hacer benchmarks.
option(NEURALSUITE_NATIVE_ARCH "Optimizar para la CPU local (-march=native)" OFF)
option(NEURALSUITE_FAST_MATH "Relajar las garantias IEEE-754 (-ffast-math)" OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
    if(NEURALSUITE_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
    if(NEURALSUITE_FAST_MATH)
        add_compile_options(-ffast-math)
    endif()
endif()

if(NEURALSUITE_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP encontrado: ejecucion paralela activada.")
    else()
        message(STATUS "OpenMP no disponible: se compila la version secuencial.")
    endif()
endif()

include_directories(include)

file(GLOB_RECURSE NEURALSUITE_SOURCES "src/*.cpp")

# Bibliotecas Compartida (.so / .dylib / .dll) y Estática (.a / .lib)
add_library(neuralsuite_shared SHARED ${NEURALSUITE_SOURCES})
add_library(neuralsuite_static STATIC ${NEURALSUITE_SOURCES})

set_target_properties(neuralsuite_shared PROPERTIES OUTPUT_NAME "neuralsuite")
set_target_properties(neuralsuite_static PROPERTIES OUTPUT_NAME "neuralsuite")

if(OpenMP_CXX_FOUND)
    target_link_libraries(neuralsuite_shared PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(neuralsuite_static PRIVATE OpenMP::OpenMP_CXX)
endif()

# Ejecutables: pruebas, demos y herramientas.
# Toda fuente ejecutable del repositorio debe estar registrada aquí; CMake es la
# única fuente de verdad del build. Nada debe compilarse a mano.
set(NEURALSUITE_EXECUTABLES
    test_suite
    demo_adaline
    demo_mlp
    demo_cnn
    demo_lstm
    demo_autoencoder
    demo_diffusion
    demo_gan
    demo_gnn
    demo_ocr
    demo_resnet
    demo_sequential
    train_llm
    generate_llm
    ocr_cli)

foreach(exe ${NEURALSUITE_EXECUTABLES})
  add_executable(${exe} ${exe}.cpp)
  target_link_libraries(${exe} PRIVATE neuralsuite_static)
endforeach()

# Binarios de las pruebas de paridad contra PyTorch (ver tools/parity/README.md).
foreach(exe parity_gpt parity_lstm)
  add_executable(${exe} tools/parity/${exe}.cpp)
  target_link_libraries(${exe} PRIVATE neuralsuite_static)
endforeach()

# `ctest` ejecuta la suite numerica. Se registran los dos modos de uso que debe
# soportar test_suite: devolver codigo de salida y no depender del directorio.
enable_testing()
add_test(NAME test_suite COMMAND test_suite)

# Reglas de Instalación como Biblioteca de C++
install(TARGETS neuralsuite_shared neuralsuite_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include/neuralsuite)
