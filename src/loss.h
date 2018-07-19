#include <cstdint>
#include <vector>

#include "ndarray.h"

namespace litecnn {

double SoftmaxLoss(const Ndarray& x, const int64_t* y, Ndarray* dx);

}
