// TODO
// * conv
// * pool
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

#endif  // LAYERS_H
