#include <cstdint>
#include <vector>

#include "ndarray.h"

double SoftmaxLoss(const Ndarray& x, const int64_t* y, Ndarray* dx);
