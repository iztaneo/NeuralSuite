cmake_minimum_required(VERSION 3.14)
project(NeuralSuite VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Banderas de optimización y OpenMP
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -march=native -ffast-math)
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

# Ejecutables Demo
add_executable(test_suite test_suite.cpp)
target_link_libraries(test_suite PRIVATE neuralsuite_static)

add_executable(demo_adaline demo_adaline.cpp)
target_link_libraries(demo_adaline PRIVATE neuralsuite_static)

add_executable(demo_mlp demo_mlp.cpp)
target_link_libraries(demo_mlp PRIVATE neuralsuite_static)

add_executable(demo_cnn demo_cnn.cpp)
target_link_libraries(demo_cnn PRIVATE neuralsuite_static)

add_executable(demo_lstm demo_lstm.cpp)
target_link_libraries(demo_lstm PRIVATE neuralsuite_static)

add_executable(train_llm train_llm.cpp)
target_link_libraries(train_llm PRIVATE neuralsuite_static)

add_executable(generate_llm generate_llm.cpp)
target_link_libraries(generate_llm PRIVATE neuralsuite_static)

# Reglas de Instalación como Biblioteca de C++
install(TARGETS neuralsuite_shared neuralsuite_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include/neuralsuite)
