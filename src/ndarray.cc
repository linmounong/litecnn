#include "ndarray.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

Ndarray::Ndarray(int64_t s0, int64_t s1, int64_t s2, int64_t s3)
    : Ndarray(std::vector<int64_t>{s0, s1, s2, s3}, nullptr) {}

Ndarray::Ndarray(const std::vector<int64_t>& shape,
                 const std::vector<float>& data)
    : Ndarray(shape, std::make_shared<std::vector<float>>(data)) {}

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
  for (int64_t i0 = 0; i0 < shape_[0]; i0++) {
    for (int64_t i1 = 0; i1 < shape_[1]; i1++) {
      for (int64_t i2 = 0; i2 < shape_[2]; i2++) {
        for (int64_t i3 = 0; i3 < shape_[3]; i3++) {
          if (at(i0, i1, i2, i3) != rhs.at(i0, i1, i2, i3)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

Ndarray Ndarray::binop(const Ndarray& rhs,
                       std::function<float(float, float)> op) const {
  Ndarray a = T();
  Ndarray b = rhs.T();
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < std::max(a.ndim(), b.ndim()); i++) {
    auto si = i < a.ndim() ? a.shape(i) : 1;
    auto sj = i < b.ndim() ? b.shape(i) : 1;
    assert(si == 1 || sj == 1 || si == sj);
    shape.push_back(std::max(si, sj));
  }
  Ndarray ret(shape, nullptr);
  for (int64_t i0 = 0; i0 < ret.shape_[0]; i0++) {
    for (int64_t i1 = 0; i1 < ret.shape_[1]; i1++) {
      for (int64_t i2 = 0; i2 < ret.shape_[2]; i2++) {
        for (int64_t i3 = 0; i3 < ret.shape_[3]; i3++) {
          ret.at(i0, i1, i2, i3) = op(
              a.at(std::min(i0, a.shape_[0] - 1), std::min(i1, a.shape_[1] - 1),
                   std::min(i2, a.shape_[2] - 1),
                   std::min(i3, a.shape_[3] - 1)),
              b.at(std::min(i0, b.shape_[0] - 1), std::min(i1, b.shape_[1] - 1),
                   std::min(i2, b.shape_[2] - 1),
                   std::min(i3, b.shape_[3] - 1)));
        }
      }
    }
  }
  return ret.T();
}

Ndarray Ndarray::operator+(const Ndarray& rhs) const {
  return binop(rhs, std::plus<float>());
}
Ndarray Ndarray::operator-(const Ndarray& rhs) const {
  return binop(rhs, std::minus<float>());
}
Ndarray Ndarray::operator*(const Ndarray& rhs) const {
  return binop(rhs, std::multiplies<float>());
}
Ndarray Ndarray::operator/(const Ndarray& rhs) const {
  return binop(rhs, std::divides<float>());
}

Ndarray Ndarray::binop(float a, std::function<float(float, float)> op) const {
  auto ret = this->fork();
  for (auto& v : *ret.data_) {
    v = op(v, a);
  }
  return ret;
}

Ndarray Ndarray::operator+(float a) const {
  return binop(a, std::plus<float>());
}
Ndarray Ndarray::operator-(float a) const {
  return binop(a, std::minus<float>());
}
Ndarray Ndarray::operator*(float a) const {
  return binop(a, std::multiplies<float>());
}
Ndarray Ndarray::operator/(float a) const {
  return binop(a, std::divides<float>());
}

void Ndarray::gaussian(float a) {
  std::minstd_rand rng(1);
  std::normal_distribution<> gaussian(0, a);
  for (int64_t i = 0; i < data_->size(); i++) {
    (*data_)[i] = gaussian(rng);
  };
}

Ndarray Ndarray::dot(const Ndarray& rhs) const {
  // see
  // https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot
  if (ndim() == 1 && rhs.ndim() == 1) {
    assert(shape(0) == rhs.shape(0));
    Ndarray ret(1);
    for (int64_t i0 = 0; i0 < shape(0); i0++) {
      ret.at(0) += at(i0) * rhs.at(i0);
    }
    return ret;
  }
  if (ndim() == 2 && rhs.ndim() == 2) {
    assert(shape(1) == rhs.shape(0));
    Ndarray ret(shape(0), rhs.shape(1));
    for (int64_t i0 = 0; i0 < shape(0); i0++) {
      for (int64_t i1 = 0; i1 < shape(1); i1++) {
        for (int64_t i2 = 0; i2 < rhs.shape(1); i2++) {
          ret.at(i0, i2) += at(i0, i1) * rhs.at(i1, i2);
        }
      }
    }
    return ret;
  }
  if (ndim() == 0 || rhs.ndim() == 0) {
    if (ndim() == 0) {
      return rhs * at();
    }
    return (*this) * rhs.at();
  }
  assert(false);  // leave empty for now
}

Ndarray Ndarray::reshape(int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
  return reshape({s0, s1, s2, s3});
}

Ndarray Ndarray::reshape(const std::vector<int64_t>& shape) {
  int64_t autoshape = -1;
  int64_t size = data_->size();
  for (int64_t i = 0; i < shape.size(); i++) {
    int64_t s = shape[i];
    if (s == 0) {
      break;
    }
    if (s == -1) {
      assert(autoshape == -1);
      autoshape = i;
      continue;
    }
    assert(s > 0);
    assert(size % s == 0);
    size /= s;
  }
  if (autoshape >= 0) {
    std::vector<int64_t> newshape = shape;
    newshape[autoshape] = size;
    return Ndarray(newshape, data_);
  }
  return Ndarray(shape, data_);
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
    std::cout << "d:" << i << " shape:" << shape_[i] << " stride:" << stride_[i]
              << std::endl;
  }
  for (float v : *data_) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

float Ndarray::sum() const {
  return std::accumulate(data_->begin(), data_->end(), 0.0f);
}

std::vector<int64_t> Ndarray::shape() const {
  std::vector<int64_t> shape = shape_;
  shape.resize(ndim());
  return shape;
}

Ndarray Ndarray::sum(int64_t dim) const {
  if (dim < 0) {
    dim += ndim();
  }
  assert(dim >= 0);
  assert(dim < ndim());
  auto newshape = shape();
  newshape[dim] = 1;
  Ndarray ret(newshape, nullptr);
  for (int64_t i0 = 0; i0 < shape_[0]; i0++) {
    int64_t j0 = dim == 0 ? 0 : i0;
    for (int64_t i1 = 0; i1 < shape_[1]; i1++) {
      int64_t j1 = dim == 1 ? 0 : i1;
      for (int64_t i2 = 0; i2 < shape_[2]; i2++) {
        int64_t j2 = dim == 2 ? 0 : i2;
        for (int64_t i3 = 0; i3 < shape_[3]; i3++) {
          int64_t j3 = dim == 3 ? 0 : i3;
          ret.at(j0, j1, j2, j3) += at(i0, i1, i2, i3);
        }
      }
    }
  }
  newshape.erase(newshape.begin() + dim);
  return ret.reshape(newshape);
}

Ndarray Ndarray::as_zeros() const {
  std::vector<int64_t> shape;
  for (int i = 0; i < ndim(); i++) {
    shape.push_back(shape_[i]);
  }
  return Ndarray(shape, nullptr);
}
