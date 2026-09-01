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

# El paralelismo lo aporta include/parallel.h con std::thread, asi que OpenMP ya
# no interviene. Se uso mientras los bucles llevaban `pragma omp`, pero eso
# dejaba sin paralelizar las plataformas donde no esta disponible: en macOS con
# AppleClang los pragmas se ignoraban y todo corria en un solo hilo, con MatMul
# usando una cuarta parte de la maquina.
find_package(Threads REQUIRED)

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


include_directories(include)

file(GLOB_RECURSE NEURALSUITE_SOURCES "src/*.cpp")

# Bibliotecas Compartida (.so / .dylib / .dll) y Estática (.a / .lib)
add_library(neuralsuite_shared SHARED ${NEURALSUITE_SOURCES})
add_library(neuralsuite_static STATIC ${NEURALSUITE_SOURCES})

set_target_properties(neuralsuite_shared PROPERTIES OUTPUT_NAME "neuralsuite")
set_target_properties(neuralsuite_static PROPERTIES OUTPUT_NAME "neuralsuite")

target_link_libraries(neuralsuite_shared PRIVATE Threads::Threads)
target_link_libraries(neuralsuite_static PRIVATE Threads::Threads)

# Ejecutables: pruebas en tests/, demos en demos/, herramientas en apps/.
#
# Salen de un patron, no de una lista. La lista escrita a mano llevaba dias sin
# `train_ocr`, y es la quinta vez en este proyecto que una enumeracion manual se
# queda corta: antes fueron `demo_autograd` en el .gitignore, `train_ocr` otra
# vez, los fuentes del Makefile y los de run_parity.sh. CMake desaconseja GLOB
# porque no se entera de los archivos nuevos sin volver a configurar; aqui pesa
# mas que nadie se acuerde de tocar la lista.
file(GLOB NEURALSUITE_PROGRAM_SOURCES
     "${CMAKE_SOURCE_DIR}/tests/*.cpp"
     "${CMAKE_SOURCE_DIR}/demos/*.cpp"
     "${CMAKE_SOURCE_DIR}/apps/*.cpp")

foreach(src ${NEURALSUITE_PROGRAM_SOURCES})
  get_filename_component(exe ${src} NAME_WE)
  add_executable(${exe} ${src})
  target_link_libraries(${exe} PRIVATE neuralsuite_static)
endforeach()

# Los benchmarks responden "cuanto tarda", no "es correcto": se compilan como
# ejecutable aparte y no forman parte de ctest, cuyo criterio es binario.
add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark PRIVATE neuralsuite_static)

# Binarios de las pruebas de paridad contra PyTorch (ver tools/parity/README.md).
foreach(exe parity_gpt parity_lstm parity_bilstm parity_crnn)
  add_executable(${exe} tools/parity/${exe}.cpp)
  target_link_libraries(${exe} PRIVATE neuralsuite_static)
endforeach()

# Volcador de pixeles para comparar el decodificador de imagen contra Pillow
# (ver tools/image/README.md).
add_executable(imgdump tools/image/imgdump.cpp)
target_link_libraries(imgdump PRIVATE neuralsuite_static)

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
