// TODO
// * conv
// * batchnorm
#ifndef LAYERS_H
#define LAYERS_H

#include <algorithm>

#include "matrix.h"

class Affine {
 public:
  Affine(int64_t m, int64_t n);

  Matrix forward(const Matrix& x);
  Matrix backward(const Matrix& dout);

  Matrix w_;
  Vector b_;
  Matrix dw_;
  Vector db_;

 private:
  Matrix x_;
};

class Relu {
 public:
  Matrix forward(const Matrix& x);
  Matrix backward(const Matrix& dout);

 private:
  Matrix x_;
};

class MaxPool {
 public:
  MaxPool(int64_t h, int64_t w, int64_t s);
  Matrix forward(const Matrix& x);
  Matrix backward(const Matrix& dout);

 private:
  Matrix out_;
  Matrix x_;
  const int64_t h_;
  const int64_t w_;
  const int64_t s_;  // stride
};

#endif  // LAYERS_H
