#include "cnn.h"

#include <vector>

#include "layers.h"
#include "loss.h"
#include "ndarray.h"

SimpleConvNet::Config& SimpleConvNet::Config::validated() {
  assert(input_depth > 0);
  assert(input_height > 0);
  assert(input_width > 0);
  assert(filter_size > 0);
  assert(n_filters > 0);
  assert(n_classes > 0);
  assert(hidden_dim > 0);
  assert(weight_scale > 0);
  assert(weight_scale > 0);
  return *this;
}

SimpleConvNet::SimpleConvNet(SimpleConvNet::Config config)
    : config_(config.validated()),
      conv_(config.filter_size, config.filter_size, config.input_depth,
            config.n_filters, 1, (config.filter_size - 1) / 2),
      pool_(2, 2, 2),
      affine_(config.n_filters * (config.input_height / 2) *
                  (config.input_width / 2),
              config.hidden_dim, config.weight_scale),
      affine2_(config.hidden_dim, config.n_classes, config.weight_scale)

{}

Ndarray SimpleConvNet::forward(const Ndarray& x) {
  assert(x.ndim() == 4);
  assert(x.shape(1) == config_.input_depth);
  assert(x.shape(2) == config_.input_height);
  assert(x.shape(3) == config_.input_width);
  auto out1 = conv_.forward(x);
  auto out2 = relu_.forward(out1);
  auto out3 = pool_.forward(out2);
  out3_shape_ = out3.shape();
  auto out4 = out3.reshape(out3.shape(0), -1);
  auto out5 = affine_.forward(out4);
  auto out6 = relu2_.forward(out5);
  auto out7 = affine2_.forward(out6);
  return out7;
}

Ndarray SimpleConvNet::backward(const Ndarray& dscores) {
  auto dout6 = affine2_.backward(dscores);
  auto dout5 = relu2_.backward(dout6);
  auto dout4 = affine_.backward(dout5);
  auto dout3 = dout4.reshape(out3_shape_);
  auto dout2 = pool_.backward(dout3);
  auto dout1 = relu_.backward(dout2);
  auto dx = conv_.backward(dout1);
  return dx;
}

double SimpleConvNet::loss(const Ndarray& x, const std::vector<int64_t>& y) {
  auto scores = forward(x);
  auto dscores = scores.as_zeros();
  auto loss = SoftmaxLoss(scores, y, &dscores);
  auto dx = backward(dscores);
  // reg loss
  return loss;
}
