#pragma once

#include <cstdint>

#include "ndarray.h"

namespace litecnn {

double SoftmaxLoss(const Ndarray& x, const int64_t* y, Ndarray* dx);

}
