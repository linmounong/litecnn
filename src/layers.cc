#include <limits>

#include "layers.h"
#include "ndarray.h"

Affine::Affine(int64_t m, int64_t n) : w_(m, n), b_(n) { w_.uniform(1); }

Ndarray Affine::forward(const Ndarray& x) {
  x_ = x;
  return x.dot(w_) + b_;
}

Ndarray Affine::backward(const Ndarray& dout) {
  db_ = dout;
  while (db_.ndim() > 1) {
    db_ = db_.sum(-2);
  }
  dw_ = x_.T().dot(dout);
  return dout.dot(w_.T());
}

// modifies x
Ndarray Relu::forward(const Ndarray& x) {
  x_ = x;
  Ndarray out = x;
  for (float& v : *out.data()) {
    if (v < 0) {
      v = 0;
    }
  }
  return out;
}

// modifies dout
Ndarray Relu::backward(const Ndarray& dout) {
  Ndarray dx = dout;
  for (int64_t i0 = 0; i0 < dx.shape(0); i0++) {
    for (int64_t i1 = 0; i1 < dx.shape(1); i1++) {
      for (int64_t i2 = 0; i2 < dx.shape(2); i2++) {
        for (int64_t i3 = 0; i3 < dx.shape(3); i3++) {
          if (x_.at(i0, i1, i2, i3) <= 0) {
            dx.at(i0, i1, i2, i3) = 0;
          }
        }
      }
    }
  }
  return dx;
}

MaxPool::MaxPool(int64_t h, int64_t w, int64_t s) : h_(h), w_(w), s_(s) {}

Ndarray MaxPool::forward(const Ndarray& x) {
  assert(x.ndim() == 2);
  Ndarray out((x.shape(0) + s_ - 1) / s_, (x.shape(1) + s_ - 1) / s_);
  for (int64_t i = 0; i < out.shape(0); i++) {
    for (int64_t j = 0; j < out.shape(1); j++) {
      float v = -std::numeric_limits<float>::infinity();
      for (int64_t ii = i * s_; ii < std::min(i * s_ + h_, x.shape(0)); ii++) {
        for (int64_t jj = j * s_; jj < std::min(j * s_ + w_, x.shape(1));
             jj++) {
          v = std::max(v, x.at(ii, jj));
        }
      }
      assert(!std::isinf(v));
      out.at(i, j) = v;
    }
  }
  out_ = out;
  x_ = x;
  return out;
}

Ndarray MaxPool::backward(const Ndarray& dout) {
  Ndarray dx = Ndarray::zeros_like(x_);
  for (int64_t i = 0; i < dout.shape(0); i++) {
    for (int64_t j = 0; j < dout.shape(1); j++) {
      for (int64_t ii = i * s_; ii < std::min(i * s_ + h_, dx.shape(0)); ii++) {
        for (int64_t jj = j * s_; jj < std::min(j * s_ + w_, dx.shape(1));
             jj++) {
          if (out_.at(i, j) == x_.at(ii, jj)) {
            dx.at(ii, jj) += dout.at(i, j);
          }
        }
      }
    }
  }
  return dx;
}
