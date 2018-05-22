#include "layers.h"

#include <cmath>
#include <iostream>

#include "ndarray.h"

Affine::Affine(int64_t m, int64_t n, double scale) : w_(m, n), b_(n) {
  w_.gaussian(scale);
}

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

Ndarray Relu::forward(const Ndarray& x) {
  x_ = x;
  Ndarray out = x.fork();
  for (double& v : *out.data()) {
    if (v < 0) {
      v = 0;
    }
  }
  return out;
}

Ndarray Relu::backward(const Ndarray& dout) {
  Ndarray dx = dout.fork();
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
          double v = -std::numeric_limits<double>::infinity();
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

Conv::Conv(int64_t fh, int64_t fw, int64_t fc, int64_t fn, int64_t s, int64_t p)
    : w_(fn, fc, fh, fw),
      b_(fn),
      fh_(fh),
      fw_(fw),
      fc_(fc),
      fn_(fn),
      s_(s),
      p_(p) {
  w_.gaussian(1);
}

Ndarray Conv::forward(const Ndarray& x) {
  assert(x.ndim() == 4);
  assert(x.shape(1) == fc_);
  int64_t N = x.shape(0);
  int64_t H = x.shape(2);
  int64_t W = x.shape(3);
  int64_t H2 = 1 + (H + 2 * p_ - fh_) / s_;
  int64_t W2 = 1 + (W + 2 * p_ - fw_) / s_;
  Ndarray out(N, fn_, H2, W2);
  // out  i
  // w_   j
  // x    k
  for (int64_t i2 = 0; i2 < out.shape(2); i2++) {
    for (int64_t i3 = 0; i3 < out.shape(3); i3++) {
      for (int64_t j2 = 0, k2 = i2 * s_ - p_; k2 < H && j2 < w_.shape(2);
           j2++, k2++) {
        if (k2 < 0) {
          continue;
        }
        for (int64_t j3 = 0, k3 = i3 * s_ - p_; k3 < W && j3 < w_.shape(3);
             j3++, k3++) {
          if (k3 < 0) {
            continue;
          }
          for (int64_t i0 = 0; i0 < out.shape(0); i0++) {
            int64_t k0 = i0;
            for (int64_t i1 = 0; i1 < out.shape(1); i1++) {
              double v = b_.at(i1);
              int64_t j0 = i1;
              for (int64_t j1 = 0; j1 < w_.shape(1); j1++) {
                int64_t k1 = j1;
                v += w_.at(j0, j1, j2, j3) * x.at(k0, k1, k2, k3);
              }
              out.at(i0, i1, i2, i3) = v;
            }
          }
        }
      }
    }
  }
  x_ = x;
  return out;
}

Ndarray Conv::backward(const Ndarray& dout) {
  assert(dout.ndim() == 4);
  int64_t H = x_.shape(2);
  int64_t W = x_.shape(3);
  db_ = dout.sum(3).sum(2).sum(0);
  dw_ = w_.as_zeros();
  Ndarray dx = x_.as_zeros();
  // out  i
  // w_   j
  // x    k
  for (int64_t i2 = 0; i2 < dout.shape(2); i2++) {
    for (int64_t i3 = 0; i3 < dout.shape(3); i3++) {
      for (int64_t j2 = 0, k2 = i2 * s_ - p_; k2 < H && j2 < w_.shape(2);
           j2++, k2++) {
        if (k2 < 0) {
          continue;
        }
        for (int64_t j3 = 0, k3 = i3 * s_ - p_; k3 < W && j3 < w_.shape(3);
             j3++, k3++) {
          if (k3 < 0) {
            continue;
          }
          for (int64_t i0 = 0; i0 < dout.shape(0); i0++) {
            int64_t k0 = i0;
            for (int64_t i1 = 0; i1 < dout.shape(1); i1++) {
              double dv = dout.at(i0, i1, i2, i3);
              int64_t j0 = i1;
              for (int64_t j1 = 0; j1 < w_.shape(1); j1++) {
                int64_t k1 = j1;
                dw_.at(j0, j1, j2, j3) += dv * x_.at(k0, k1, k2, k3);
                dx.at(k0, k1, k2, k3) += dv * w_.at(j0, j1, j2, j3);
              }
            }
          }
        }
      }
    }
  }
  return dx;
}
