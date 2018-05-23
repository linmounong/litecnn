#include "loss.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "ndarray.h"

double SoftmaxLoss(const Ndarray& x, const int64_t* y, Ndarray* dx) {
  assert(dx != nullptr);
  assert(x.ndim() == 2);
  int64_t n = x.shape(0);
  int64_t c = x.shape(1);
  double loss = 0;
  for (int64_t i = 0; i < n; i++) {
    double max = x.at(i, 0);
    for (int64_t j = 1; j < c; j++) {
      double v = x.at(i, j);
      if (max < v) {
        max = v;
      }
    }
    double sum = 0;
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
