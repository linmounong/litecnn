.PHONY: all clean test

CXX = c++
CXXFLAGS = -pthread -std=c++11 -march=native
OBJS = layers.o ndarray.o loss.o

all: $(OBJS)

%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c $<

bin/unittest: src/unittest.cc $(OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o bin

test: bin/unittest
	bin/unittest
