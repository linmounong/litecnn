#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "matrix.h"

void Matrix::zero() { std::fill(data_.begin(), data_.end(), 0.0); }

void Matrix::uniform(float a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < data_.size(); i++) {
    data_[i] = uniform(rng);
  };
}

Matrix Matrix::inner(const Matrix& m, int64_t offset_i,
                     int64_t offset_j) const {
  Matrix ret = m;
  for (int64_t i = 0; i < m.rows(); i++) {
    for (int64_t j = 0; j < m.cols(); j++) {
      float v = at_or_zero(i + offset_i, j + offset_j) * m.at(i, j);
      ret.set(v, i, j);
    }
  }
  return ret;
}

Matrix Matrix::dot(const Matrix& m) const {
  assert(cols() == m.rows());
  Matrix ret(rows(), m.cols());
  for (int64_t i = 0; i < ret.rows(); i++) {
    for (int64_t j = 0; j < ret.cols(); j++) {
      float v = 0;
      for (int64_t k = 0; k < cols(); k++) {
        v += at(i, k) * m.at(k, j);
      }
      ret.set(v, i, j);
    }
  }
  return ret;
}

float Matrix::sum() const {
  return std::accumulate(data_.begin(), data_.end(), 0);
}

bool Matrix::operator==(const Matrix& m) {
  if (rows() != m.rows() || cols() != m.cols()) {
    return false;
  }
  for (int64_t i = 0; i < rows(); i++) {
    for (int64_t j = 0; j < cols(); j++) {
      if (at(i, j) != m.at(i, j)) {
        return false;
      }
    }
  }
  return true;
}

Matrix Matrix::operator+(const Matrix& m) const {
  assert(rows() == m.rows() && cols() == m.cols());
  Matrix ret = m;
  for (int64_t i = 0; i < ret.rows(); i++) {
    for (int64_t j = 0; j < ret.cols(); j++) {
      ret.set(ret.at(i, j) + at(i, j), i, j);
    }
  }
  return ret;
}

Matrix Matrix::T() const {
  Matrix ret = *this;
  ret.m_ = n_;
  ret.n_ = m_;
  ret.si_ = sj_;
  ret.sj_ = si_;
  return ret;
};
