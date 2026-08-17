CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -march=native -ffast-math -Iinclude


# Detección de Sistema Operativo
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -fopenmp
endif
ifeq ($(UNAME_S),Darwin)
    # macOS Clang
    CXXFLAGS += -Xpreprocessor -fopenmp 2>/dev/null || true
endif

SRCS = src/tensor.cpp src/tokenizer.cpp
TARGETS = test_suite demo_adaline demo_mlp demo_cnn demo_lstm train_llm generate_llm

all: $(TARGETS)

test_suite: test_suite.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

demo_adaline: demo_adaline.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

demo_mlp: demo_mlp.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^


demo_cnn: demo_cnn.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

demo_lstm: demo_lstm.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

train_llm: train_llm.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

generate_llm: generate_llm.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGETS) *.o
