// TODO
// * conv
// * relu
// * pool
// * batchnorm
#ifndef LAYERS_H
#define LAYERS_H

#include <string>

#include "matrix.h"

class AffineLayer {
 public:
  AffineLayer(int64_t m, int64_t n);

  Matrix forward(const Matrix& x);
  Matrix backward(const Matrix& dout);

 private:
  Matrix x_;
  Matrix w_;
  Matrix dw_;
  Vector b_;
  Vector db_;
};

#endif  // LAYERS_H
