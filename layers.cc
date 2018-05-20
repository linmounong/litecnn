#include "layers.h"
#include "matrix.h"

Affine::Affine(int64_t m, int64_t n) : w_(m, n), b_(n) {
  w_.uniform(1);
  b_.zero();
}

Matrix Affine::forward(const Matrix& x) {
  x_ = x;
  return x.dot(w_) + b_;
}

Matrix Affine::backward(const Matrix& dout) {
  db_ = b_;
  db_.zero();
  for (int64_t i = 0; i < dout.rows(); i++) {
    db_.add(dout, i);
  }
  dw_ = x_.T().dot(dout);
  return dout.dot(w_.T());
}

Matrix Relu::forward(const Matrix& x) {
  x_ = x;
  Matrix out = x;
  for (int64_t i = 0; i < x.rows(); i++) {
    for (int64_t j = 0; j < x.cols(); j++) {
      out.at(i, j) = std::max(x.at(i, j), 0.0f);
    }
  }
  return out;
}

Matrix Relu::backward(const Matrix& dout) {
  Matrix dx = dout;
  for (int64_t i = 0; i < dout.rows(); i++) {
    for (int64_t j = 0; j < dout.cols(); j++) {
      if (x_.at(i, j) <= 0) {
        dx.at(i, j) = 0;
      }
    }
  }
  return dx;
}
