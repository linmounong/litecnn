#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

class Matrix {
 public:
  Matrix();
  Matrix(int64_t n, int64_t m);
  // for testing
  explicit Matrix(const std::vector<std::vector<float>>& m);

  inline float at(int64_t i, int64_t j) const {
    assert(i >= 0 && i < n_ && j >= 0 && j < m_);
    return data_[i * si_ + j * sj_];
  };

  inline float set(float v, int64_t i, int64_t j) {
    assert(i >= 0 && i < n_ && j >= 0 && j < m_);
    data_[i * si_ + j * sj_] = v;
  };

  inline float at_or_zero(int64_t i, int64_t j) const {
    if (i < 0 || i >= n_ || j < 0 || j >= m_) {
      return 0;
    }
    return at(i, j);
  };

  inline int64_t cols() const { return m_; }

  inline int64_t rows() const { return n_; }

  void zero();

  void uniform(float a);

  bool operator==(const Matrix& m);

  Matrix operator+(const Matrix& m) const;

  // takes shape of m, padding with zero
  Matrix inner(const Matrix& m, int64_t offset_i = 0,
               int64_t offset_j = 0) const;

  Matrix dot(const Matrix& m) const;

  Matrix T() const;

  float sum() const;

 private:
  int64_t n_;
  int64_t m_;
  int64_t si_;
  int64_t sj_;
  std::vector<float> data_;
};

#endif  // MATRIX_H
