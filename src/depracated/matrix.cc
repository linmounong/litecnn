#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "matrix.h"

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(int64_t n, int64_t m)
    : n_(n), m_(m), si_(m), sj_(1), data_(n * m) {}

Matrix::Matrix(const std::vector<std::vector<float>>& m)
    : Matrix(m.size(), m[0].size()) {
  for (int64_t i = 0; i < m.size(); i++) {
    for (int64_t j = 0; j < m[0].size(); j++) {
      at(i, j) = m[i][j];
    }
  }
}

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
      ret.at(i, j) *= at_or_zero(i + offset_i, j + offset_j);
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
      ret.at(i, j) = v;
    }
  }
  return ret;
}

float Matrix::sum() const {
  return std::accumulate(data_.begin(), data_.end(), 0);
}

bool Matrix::operator==(const Matrix& m) const {
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
      ret.at(i, j) += at(i, j);
    }
  }
  return ret;
}

Matrix Matrix::operator+(const Vector& v) const {
  assert(cols() == v.size());
  Matrix ret = *this;
  for (int64_t i = 0; i < rows(); i++) {
    for (int64_t j = 0; j < cols(); j++) {
      ret.at(i, j) += v[j];
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

void Matrix::debug() const {
  for (int64_t i = 0; i < rows(); i++) {
    for (int64_t j = 0; j < cols(); j++) {
      std::cout << at(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

Vector::Vector() : Vector(0) {}
Vector::Vector(int64_t n) : data_(n) {}
Vector::Vector(const std::vector<float>& v) : data_(v) {}

void Vector::zero() { std::fill(data_.begin(), data_.end(), 0.0); }

void Vector::add(const Matrix& m, int64_t i) {
  assert(i >= 0);
  assert(i < m.rows());
  assert(size() == m.cols());
  for (int64_t j = 0; j < m.cols(); j++) {
    data_[j] += m.at(i, j);
  }
}

bool Vector::operator==(const Vector& v) const { return data_ == v.data_; }
