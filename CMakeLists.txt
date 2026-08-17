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

# Banderas de optimización y OpenMP
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -march=native -ffast-math)
    find_package(OpenMP REQUIRED)
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

# Reglas de Instalación como Biblioteca de C++
install(TARGETS neuralsuite_shared neuralsuite_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include/neuralsuite)
