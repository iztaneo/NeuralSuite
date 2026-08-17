CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -fPIC -march=native -ffast-math -Iinclude -Wl,-rpath,'$$ORIGIN'


# Detección de Sistema Operativo
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -fopenmp
    LIB_EXT := .so
endif
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -Xpreprocessor -fopenmp 2>/dev/null || true
    LIB_EXT := .dylib
endif

SRCS = src/tensor.cpp src/tokenizer.cpp
OBJS = $(SRCS:.cpp=.o)

LIB_STATIC = libneuralsuite.a
LIB_SHARED = libneuralsuite$(LIB_EXT)

TARGETS = $(LIB_STATIC) $(LIB_SHARED) test_suite demo_adaline demo_mlp demo_sequential demo_autoencoder demo_resnet demo_gan demo_gnn demo_diffusion demo_ocr demo_ocr_mitsubishi ocr_cli process_book_page demo_cnn demo_lstm train_llm generate_llm


all: $(TARGETS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_STATIC): $(OBJS)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJS)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^

test_suite: test_suite.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_adaline: demo_adaline.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_mlp: demo_mlp.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_sequential: demo_sequential.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_autoencoder: demo_autoencoder.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_resnet: demo_resnet.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_gan: demo_gan.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_gnn: demo_gnn.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_diffusion: demo_diffusion.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_ocr: demo_ocr.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_ocr_mitsubishi: demo_ocr_mitsubishi.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

ocr_cli: ocr_cli.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

process_book_page: process_book_page.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_cnn: demo_cnn.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

demo_lstm: demo_lstm.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

train_llm: train_llm.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

generate_llm: generate_llm.cpp $(LIB_STATIC)
	$(CXX) $(CXXFLAGS) -o $@ $< -L. -lneuralsuite

clean:
	rm -f test_suite demo_adaline demo_mlp demo_cnn demo_lstm train_llm generate_llm $(LIB_STATIC) $(LIB_SHARED) src/*.o

.PHONY: all clean
