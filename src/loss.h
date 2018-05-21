#include <cstdint>
#include <vector>

#include "ndarray.h"

float SoftmaxLoss(const Ndarray& x, const std::vector<int64_t>& y, Ndarray* dx);
