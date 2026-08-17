cmake_minimum_required(VERSION 3.14)
project(NeuralSuite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Directorios de inclusión
include_directories(include)

# Buscar OpenMP opcionalmente para aceleración multi-hilo
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado: Activando paralelismo multi-hilo en CPU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Banderas de optimización del compilador
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Archivos fuente compartidos de la biblioteca
set(SUITE_SOURCES
    src/tensor.cpp
    src/tokenizer.cpp
)

# Binarios ejecutables
add_executable(test_suite test_suite.cpp ${SUITE_SOURCES})
add_executable(demo_mlp demo_mlp.cpp ${SUITE_SOURCES})
add_executable(demo_cnn demo_cnn.cpp ${SUITE_SOURCES})
add_executable(demo_lstm demo_lstm.cpp ${SUITE_SOURCES})
add_executable(train_llm train_llm.cpp ${SUITE_SOURCES})
add_executable(generate_llm generate_llm.cpp ${SUITE_SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_suite PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(demo_mlp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(demo_cnn PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(demo_lstm PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(train_llm PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(generate_llm PRIVATE OpenMP::OpenMP_CXX)
endif()
