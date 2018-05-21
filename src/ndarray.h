#ifndef NDARRAY_H
#define NDARRAY_H

#include <cassert>
#include <cstdint>
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

  inline float at(const std::vector<int64_t>& idx) const {
    int64_t n = 0;
    for (int64_t i = 0; i < idx.size(); i++) {
      n += idx[i] * stride_[i];
    }
    return (*data_)[n];
  };

  inline float& at(const std::vector<int64_t>& idx) {
    int64_t n = 0;
    for (int64_t i = 0; i < idx.size(); i++) {
      n += idx[i] * stride_[i];
    }
    return (*data_)[n];
  };

  inline int64_t ndim() const { return ndim_; }

  inline std::vector<float>* data() const { return data_.get(); }

  inline int64_t shape(int64_t dim) const {
    if (dim < 0) {
      dim += ndim();
    }
    assert(dim >= 0);
    assert(dim < shape_.size());
    return shape_[dim];
  }

  std::vector<int64_t> shape() const;

  bool operator==(const Ndarray& rhs) const;

  // broadcast
  Ndarray operator+(const Ndarray& rhs) const;

  Ndarray operator*(float a) const;

  Ndarray reshape(int64_t s0 = 0, int64_t s1 = 0, int64_t s2 = 0,
                  int64_t s3 = 0);

  Ndarray reshape(const std::vector<int64_t>& shape);

  Ndarray T() const;

  void zero();

  float sum() const;

  Ndarray sum(int64_t dim) const;

  void gaussian(float a);

  Ndarray fork() const;

  Ndarray dot(const Ndarray& rhs) const;

  Ndarray as_zeros() const;

  void debug() const;

 private:
  int64_t ndim_ = 0;
  std::shared_ptr<std::vector<float>> data_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
};

#endif  // NDARRAY_H
