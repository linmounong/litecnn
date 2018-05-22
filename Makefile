.PHONY: all clean test

CXX = c++
FLAGS = -pthread -std=c++11 -march=native -O3 ${CXXFLAGS}
OBJS = layers.o ndarray.o loss.o cnn.o

all: $(OBJS)

%.o: src/%.cc
	$(CXX) $(FLAGS) -c $<

bin/unittest: src/unittest.cc $(OBJS)
	@mkdir -p bin
	$(CXX) $(FLAGS) -o $@ $^

clean:
	rm -rf *.o bin

test: bin/unittest
	bin/unittest
