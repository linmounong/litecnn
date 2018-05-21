#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "ndarray.h"

Ndarray::Ndarray(int64_t s0, int64_t s1, int64_t s2, int64_t s3)
    : Ndarray(std::vector<int64_t>{s0, s1, s2, s3}, nullptr) {}

Ndarray::Ndarray(const std::vector<int64_t>& shape,
                 std::shared_ptr<std::vector<float>> data)
    : shape_(4, 1), stride_(4, 1) {
  assert(shape.size() <= 4);
  int64_t size = 1;
  for (int64_t s : shape) {
    if (s <= 0) {
      break;
    }
    size *= s;
    shape_[ndim_] = s;
    ndim_ += 1;
  }
  if (data) {
    assert(data->size() == size);
    data_ = data;
  } else {
    data_ = std::make_shared<std::vector<float>>(size);
  }
  for (int64_t stride = 1, i = ndim_ - 1; i >= 0; i--) {
    stride_[i] = stride;
    stride *= shape_[i];
  }
}

bool Ndarray::operator==(const Ndarray& rhs) const {
  if (ndim() != rhs.ndim() || shape_ != rhs.shape_) {
    return false;
  }
  for (int64_t i0 = 0; i0 < shape(0); i0++) {
    for (int64_t i1 = 0; i1 < shape(1); i1++) {
      for (int64_t i2 = 0; i2 < shape(2); i2++) {
        for (int64_t i3 = 0; i3 < shape(3); i3++) {
          if (at(i0, i1, i2, i3) != rhs.at(i0, i1, i2, i3)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void Ndarray::uniform(float a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < data_->size(); i++) {
    (*data_)[i] = uniform(rng);
  };
}

/*
Ndarray Ndarray::dot(const Ndarray& m) const {
  assert(cols() == m.rows());
  Ndarray ret(rows(), m.cols());
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
*/

Ndarray Ndarray::reshape(int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
  return Ndarray({s0, s1, s2, s3}, data_);
}

Ndarray Ndarray::T() const {
  Ndarray ret = *this;
  for (int64_t i = 0, j = ndim() - 1; i < j; i++, j--) {
    int64_t tmp = ret.shape_[i];
    ret.shape_[i] = ret.shape_[j];
    ret.shape_[j] = tmp;
    tmp = ret.stride_[i];
    ret.stride_[i] = ret.stride_[j];
    ret.stride_[j] = tmp;
  }
  return ret;
};

Ndarray Ndarray::fork() const {
  Ndarray ret = *this;
  ret.data_ = std::make_shared<std::vector<float>>(*data_);
  return ret;
}

void Ndarray::debug() const {
  std::cout << "ndim:" << ndim() << std::endl;
  for (int64_t i = 0; i < ndim(); i++) {
    std::cout << "shape:" << shape_[i] << " stride:" << stride_[i] << std::endl;
  }
}
