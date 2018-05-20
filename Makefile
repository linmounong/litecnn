.PHONY: all clean test

CXX = c++
CXXFLAGS = -pthread -std=c++11 -march=native
OBJS = matrix.o
TESTS = matrix_test

all: $(OBJS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

%_test: %_test.cc $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ && ./$@

clean:
	rm -f *.o *_test

test: $(TESTS)
	@echo "all passed"
