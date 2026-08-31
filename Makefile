# Copyright 2026 NeuralSuite Authors.
# Licensed under the Apache License, Version 2.0.
#
# Build alternativo al de CMake, para quien solo quiera `make`. CMakeLists.txt
# sigue siendo el que usa la integracion continua; los dos deben producir
# binarios equivalentes, asi que las banderas se mantienen alineadas.

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -fPIC -Iinclude

# El paralelismo lo aporta include/parallel.h con std::thread, asi que aqui solo
# hace falta -pthread. Antes esta seccion anadia -fopenmp, y en macOS la linea
# terminaba en `2>/dev/null || true`, que make pasaba al compilador como si
# fueran ficheros: la compilacion no producia ningun objeto y `make` fallaba.
CXXFLAGS += -pthread

# -MMD -MP hace que el compilador escriba, junto a cada objeto, la lista de
# cabeceras de las que depende. Sin esto `make` solo miraba el .cpp: tocar un
# archivo de include/ no recompilaba nada y quedaba un binario construido con la
# version anterior de la cabecera. Es un fallo especialmente traicionero porque
# no da ningun error, solo resultados que no corresponden al codigo que se esta
# leyendo.
CXXFLAGS += -MMD -MP

# Dos banderas que estaban siempre puestas y ahora hay que pedir:
#
#   NATIVE=1     -march=native, que produce binarios que pueden no arrancar en
#                otra CPU. Util para medir en la maquina propia.
#   FAST_MATH=1  -ffast-math, que relaja IEEE-754 (NaN, infinitos,
#                asociatividad) justo donde se verifican gradientes.
#
# Son las mismas opciones que CMakeLists.txt expone como NEURALSUITE_NATIVE_ARCH
# y NEURALSUITE_FAST_MATH, tambien desactivadas por defecto. Cuando estaban
# fijas aqui, este build y el de CI no calculaban lo mismo.
ifeq ($(NATIVE),1)
    CXXFLAGS += -march=native
endif
ifeq ($(FAST_MATH),1)
    CXXFLAGS += -ffast-math
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := .dylib
else
    LIB_EXT := .so
endif

# Los fuentes salen de un patron, no de una lista escrita a mano. Esa lista ya
# se olvido de demo_autograd y de train_ocr, las dos veces por lo mismo: hay que
# acordarse de tocarla. CMake usa GLOB_RECURSE por la misma razon.
SRCS = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)
OBJS = $(SRCS:.cpp=.o)

LIB_STATIC = libneuralsuite.a
LIB_SHARED = libneuralsuite$(LIB_EXT)

# Los ejecutables son todos los .cpp de la raiz, sin enumerarlos. Antes habia
# tres listas que mantener a mano -la de objetivos, una regla por programa y la
# de `clean`-, y las tres habian divergido: demo_autograd no se compilaba y
# `clean` solo borraba cuatro de los quince binarios.
PROGRAMS = $(patsubst %.cpp,%,$(wildcard *.cpp))
BENCHMARK = benchmarks/benchmark

TARGETS = $(LIB_STATIC) $(LIB_SHARED) $(PROGRAMS)

all: $(TARGETS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_STATIC): $(OBJS)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJS)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^

# Una sola regla para todos: el patron cubre los demos, las herramientas y la
# suite de pruebas, que se compilan exactamente igual.
#
# Se enlaza la biblioteca estatica por su ruta, y no con `-L. -lneuralsuite`.
# Con las dos versiones presentes el enlazador prefiere la dinamica, y entonces
# el binario deja de funcionar en cuanto se ejecuta desde otro directorio: no
# encuentra el .dylib. Habia un -Wl,-rpath,'$$ORIGIN' que pretendia evitarlo,
# pero `$$ORIGIN` es sintaxis de Linux y en macOS no significa nada -alli se
# escribe @loader_path-, de modo que no servia en la unica plataforma donde el
# problema se daba. Enlazando en estatico el binario no depende de nada y se
# puede copiar a cualquier sitio.
%: %.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_STATIC)

# Fuera de `all`: medir con -O3 generico no dice mucho, conviene NATIVE=1.
benchmark: benchmarks/benchmark.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $(BENCHMARK) $< $(LIB_STATIC)

test: test_suite
	./test_suite

# Las dependencias que genera el compilador se incluyen si existen. El guion
# evita que `make` falle la primera vez, cuando todavia no hay ninguna.
-include $(OBJS:.o=.d) $(PROGRAMS:=.d) $(BENCHMARK).d

clean:
	rm -f $(PROGRAMS) $(BENCHMARK) $(LIB_STATIC) $(LIB_SHARED) src/*.o src/*.d *.d benchmarks/*.d

.PHONY: all clean test benchmark
