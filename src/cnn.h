#include <vector>

#include "layers.h"
#include "loss.h"
#include "ndarray.h"

// conv - relu - 2x2 pool - affine - relu - affine - softmax
class SimpleConvNet {
 public:
  struct Config {
    int64_t input_height = 0;
    int64_t input_width = 0;
    int64_t input_depth = 0;
    int64_t n_filters = 0;
    int64_t filter_size = 0;
    int64_t hidden_dim = 0;
    float weight_scale = 0;
    int64_t n_classes = 0;
    float reg = 0;

    Config& validated();
  };

  SimpleConvNet(Config config);

  float loss(const Ndarray& x, const std::vector<int64_t>& y);

  Ndarray forward(const Ndarray& x);
  Ndarray backward(const Ndarray& dscores);

  // layers
  Conv conv_;
  Relu relu_;
  MaxPool pool_;
  Affine affine_;
  Relu relu2_;
  Affine affine2_;

 private:
  Config config_;
  std::vector<int64_t> out3_shape_;
};
