#ifndef VECTOR_H
#define VECTOR_H

#include <cstdint>
#include <ostream>
#include <vector>

class Vector {
 public:
  explicit Vector(int64_t n);

  inline float& operator[](int64_t i) { return data_[i]; }
  inline const float& operator[](int64_t i) const { return data_[i]; }
  inline int64_t size() const { return data_.size(); }

  void zero();

 private:
  std::vector<float> data_;
};

#endif  // VECTOR_H
