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
  Ndarray xt = x.T();
  auto shape = xt.shape();
  shape[0] = (xt.shape(0) + s_ - 1) / s_;
  shape[1] = (xt.shape(1) + s_ - 1) / s_;
  Ndarray outt(shape, nullptr);
  for (int64_t i0 = 0; i0 < outt.shape(0); i0++) {
    for (int64_t i1 = 0; i1 < outt.shape(1); i1++) {
      for (int64_t i2 = 0; i2 < outt.shape(2); i2++) {
        for (int64_t i3 = 0; i3 < outt.shape(3); i3++) {
          float v = -std::numeric_limits<float>::infinity();
          for (int64_t ii = i0 * s_; ii < std::min(i0 * s_ + w_, xt.shape(0));
               ii++) {
            for (int64_t jj = i1 * s_; jj < std::min(i1 * s_ + h_, xt.shape(1));
                 jj++) {
              v = std::max(v, xt.at(ii, jj, i2, i3));
            }
          }
          assert(!std::isinf(v));
          outt.at(i0, i1, i2, i3) = v;
        }
      }
    }
  }
  outt_ = outt;
  xt_ = xt;
  return outt.T();
}

Ndarray MaxPool::backward(const Ndarray& dout) {
  Ndarray doutt = dout.T();
  Ndarray dxt = xt_.as_zeros();
  for (int64_t i0 = 0; i0 < doutt.shape(0); i0++) {
    for (int64_t i1 = 0; i1 < doutt.shape(1); i1++) {
      for (int64_t i2 = 0; i2 < doutt.shape(2); i2++) {
        for (int64_t i3 = 0; i3 < doutt.shape(3); i3++) {
          for (int64_t ii = i0 * s_; ii < std::min(i0 * s_ + w_, dxt.shape(0));
               ii++) {
            for (int64_t jj = i1 * s_;
                 jj < std::min(i1 * s_ + h_, dxt.shape(1)); jj++) {
              if (outt_.at(i0, i1, i2, i3) == xt_.at(ii, jj, i2, i3)) {
                dxt.at(ii, jj, i2, i3) += doutt.at(i0, i1, i2, i3);
              }
            }
          }
        }
      }
    }
  }
  return dxt.T();
}
