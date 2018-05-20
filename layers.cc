#include "layers.h"
#include "matrix.h"

AffineLayer::AffineLayer(int64_t m, int64_t n) : w_(m, n), b_(n) {
  w_.uniform(1);
  b_.zero();
}

Matrix AffineLayer::forward(const Matrix& x) {
  x_ = x;
  return x.dot(w_) + b_;
}

Matrix AffineLayer::backward(const Matrix& dout) {
  db_ = b_;
  db_.zero();
  for (int64_t i = 0; i < dout.rows(); i++) {
    db_.add(dout, i);
  }
  dw_ = x_.T().dot(dout);
  return dout.dot(w_.T());
}
