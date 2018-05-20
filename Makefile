.PHONY: all clean test

CXX = c++
CXXFLAGS = -pthread -std=c++11 -march=native
OBJS = matrix.o layers.o

all: $(OBJS)

%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c $<

bin/unittest: unittest.cc $(OBJS)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o bin

test: bin/unittest
	bin/unittest
