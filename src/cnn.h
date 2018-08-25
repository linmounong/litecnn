#include <atomic>
#include <vector>

#include "layers.h"
#include "loss.h"
#include "ndarray.h"

namespace litecnn {

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
    double weight_scale = 0;
    int64_t n_classes = 0;

    // ftrl
    double alpha = 5;
    double beta = 0.001;
    double lambda1 = 0.00;
    double lambda2 = 0.0;

    Config& validated();
  };

  explicit SimpleConvNet(Config config);

  double loss(const Ndarray& x, const int64_t* y);

  Ndarray forward(const Ndarray& x);
  Ndarray backward(const Ndarray& dscores);

  // thread safe
  void train(const Ndarray& x, const int64_t* y, const Ndarray& x_val,
             const int64_t* y_val, int epochs, int64_t batch, double lr,
             int64_t log_every, int64_t eval_every);

  void predict(const Ndarray& x, int64_t* y);

  double eval(const Ndarray& x, const int64_t* y);

  SimpleConvNet snapshot();

  // layers
  Conv conv_;
  Relu relu_;
  MaxPool pool_;
  Affine affine_;
  Relu relu2_;
  Affine affine2_;

 private:
  Config config_;
  std::vector<int64_t> shape_before_affine_;

  std::shared_ptr<std::atomic_int> iter_;
};

}  // namespace litecnn
