.PHONY: all clean test data train

CXX = c++
FLAGS = -pthread -std=c++11 -march=native -O3 ${CXXFLAGS} -I third_party/mnist/include
OBJS = layers.o ndarray.o loss.o cnn.o
BINS = bin/unittest bin/mnist_main

all: $(OBJS) $(BINS)

%.o: src/%.cc
	$(CXX) $(FLAGS) -c $<

bin/%: src/%.cc $(OBJS)
	@mkdir -p bin
	$(CXX) $(FLAGS) -o $@ $^

clean:
	rm -rf *.o bin

test: bin/unittest
	bin/unittest

data: mnist/train-images-idx3-ubyte mnist/train-labels-idx1-ubyte mnist/t10k-images-idx3-ubyte mnist/t10k-labels-idx1-ubyte

mnist/%:
	@mkdir -p mnist
	wget -O $@.gz http://yann.lecun.com/exdb/$@.gz && gunzip $@.gz


train: bin/mnist_main data
	bin/mnist_main
