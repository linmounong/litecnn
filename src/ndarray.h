#ifndef NDARRAY_H
#define NDARRAY_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

class Ndarray {
 public:
  Ndarray(int64_t s0 = 0, int64_t s1 = 0, int64_t s2 = 0, int64_t s3 = 0);
  // for testing
  Ndarray(const std::vector<int64_t>& shape, const std::vector<float>& data);
  Ndarray(const std::vector<int64_t>& shape,
          std::shared_ptr<std::vector<float>> data);

  inline float at(int64_t i = 0, int64_t j = 0, int64_t k = 0,
                  int64_t l = 0) const {
    assert(i >= 0);
    assert(i < shape_[0]);
    assert(j >= 0);
    assert(j < shape_[1]);
    assert(k >= 0);
    assert(k < shape_[2]);
    assert(l >= 0);
    assert(l < shape_[3]);
    return (*data_)[i * stride_[0] + j * stride_[1] + k * stride_[2] +
                    l * stride_[3]];
  };

  inline float& at(int64_t i = 0, int64_t j = 0, int64_t k = 0, int64_t l = 0) {
    assert(i >= 0);
    assert(i < shape_[0]);
    assert(j >= 0);
    assert(j < shape_[1]);
    assert(k >= 0);
    assert(k < shape_[2]);
    assert(l >= 0);
    assert(l < shape_[3]);
    return (*data_)[i * stride_[0] + j * stride_[1] + k * stride_[2] +
                    l * stride_[3]];
  };

  inline int64_t ndim() const { return ndim_; }

  inline std::vector<float>* data() const { return data_.get(); }

  inline int64_t shape(int64_t dim) const {
    assert(dim >= 0 && dim < ndim());
    return shape_[dim];
  }

  bool operator==(const Ndarray& rhs) const;

  // broadcast
  Ndarray operator+(const Ndarray& rhs) const;

  Ndarray operator*(float a) const;

  Ndarray reshape(int64_t s0 = 0, int64_t s1 = 0, int64_t s2 = 0,
                  int64_t s3 = 0);

  Ndarray T() const;

  void zero();

  float sum() const;

  void uniform(float a);

  Ndarray fork() const;

  Ndarray dot(const Ndarray& rhs) const;

  void debug() const;

 private:
  int64_t ndim_ = 0;
  std::shared_ptr<std::vector<float>> data_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
};

#endif  // NDARRAY_H
