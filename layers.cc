#include <limits>

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

MaxPool::MaxPool(int64_t h, int64_t w, int64_t s) : h_(h), w_(w), s_(s) {}

Matrix MaxPool::forward(const Matrix& x) {
  Matrix out((x.rows() + s_ - 1) / s_, (x.cols() + s_ - 1) / s_);
  for (int64_t i = 0; i < out.rows(); i++) {
    for (int64_t j = 0; j < out.cols(); j++) {
      float v = -std::numeric_limits<float>::infinity();
      for (int64_t ii = i * s_; ii < std::min(i * s_ + h_, x.rows()); ii++) {
        for (int64_t jj = j * s_; jj < std::min(j * s_ + w_, x.cols()); jj++) {
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

Matrix MaxPool::backward(const Matrix& dout) {
  Matrix dx = x_;
  dx.zero();
  for (int64_t i = 0; i < dout.rows(); i++) {
    for (int64_t j = 0; j < dout.cols(); j++) {
      for (int64_t ii = i * s_; ii < std::min(i * s_ + h_, dx.rows()); ii++) {
        for (int64_t jj = j * s_; jj < std::min(j * s_ + w_, dx.cols()); jj++) {
          if (out_.at(i, j) == x_.at(ii, jj)) {
            dx.at(ii, jj) += dout.at(i, j);
          }
        }
      }
    }
  }
  return dx;
}
