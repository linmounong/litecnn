// TODO
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
  Ndarray outt_;
  Ndarray xt_;
  const int64_t h_;
  const int64_t w_;
  const int64_t s_;  // stride
};

class Conv {
 public:
  Conv(int64_t fh, int64_t fw, int64_t fc, int64_t fn, int64_t s, int64_t p);
  Ndarray forward(const Ndarray& x);      // N,fc,H,W
  Ndarray backward(const Ndarray& dout);  // N,fn,H',W'

  Ndarray w_;   // (fn,fc,fh,fw)
  Ndarray dw_;  // (fn,fc,fh,fw)
  Ndarray b_;   // (fc,)
  Ndarray db_;  // (fc,)

 private:
  const int64_t fh_;  // filter height
  const int64_t fw_;  // filter width
  const int64_t fc_;  // filter depth
  const int64_t fn_;  // number of filters
  const int64_t s_;   // stride
  const int64_t p_;   // stride
  Ndarray x_;
};

#endif  // LAYERS_H
