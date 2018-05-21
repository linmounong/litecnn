#include "loss.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "ndarray.h"

float SoftmaxLoss(const Ndarray& x, const std::vector<int64_t>& y,
                  Ndarray* dx) {
  assert(dx != nullptr);
  assert(x.ndim() == 2);
  int64_t n = x.shape(0);
  int64_t c = x.shape(1);
  assert(n == y.size());
  float loss = 0;
  for (int64_t i = 0; i < n; i++) {
    float max = x.at(i, 0);
    for (int64_t j = 1; j < c; j++) {
      float v = x.at(i, j);
      if (max < v) {
        max = v;
      }
    }
    float sum = 0;
    for (int64_t j = 0; j < c; j++) {
      sum += dx->at(i, j) = std::exp(x.at(i, j) - max);
    }
    for (int64_t j = 0; j < c; j++) {
      dx->at(i, j) /= sum;
      if (j == y[i]) {
        loss -= std::log(dx->at(i, j));
        dx->at(i, j) -= 1;
      }
    }
  }
  *dx = *dx * (1.0 / n);
  loss /= n;
  return loss;
}
