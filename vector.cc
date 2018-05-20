#include "vector.h"

#include <algorithm>
#include <cassert>
#include <cmath>

Vector::Vector(int64_t n) : data_(n) {}

void Vector::zero() { std::fill(data_.begin(), data_.end(), 0.0); }
