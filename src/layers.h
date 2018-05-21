// TODO
// * conv
// * batchnorm
#ifndef LAYERS_H
#define LAYERS_H

#include <algorithm>

#include "ndarray.h"

class Affine {
 public:
  Affine(int64_t m, int64_t n);

  Ndarray forward(const Ndarray& x);
  Ndarray backward(const Ndarray& dout);

  Ndarray w_;
  Ndarray b_;
  Ndarray dw_;
  Ndarray db_;

 private:
  Ndarray x_;
};

class Relu {
 public:
  Ndarray forward(const Ndarray& x);
  Ndarray backward(const Ndarray& dout);

 private:
  Ndarray x_;
};

class MaxPool {
 public:
  MaxPool(int64_t h, int64_t w, int64_t s);
  Ndarray forward(const Ndarray& x);
  Ndarray backward(const Ndarray& dout);

 private:
  Ndarray out_;
  Ndarray x_;
  const int64_t h_;
  const int64_t w_;
  const int64_t s_;  // stride
};

#endif  // LAYERS_H
