#include "cnn.h"

#include <iostream>
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
            config.n_filters, 1, (config.filter_size - 1) / 2,
            config.weight_scale),
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

double SimpleConvNet::loss(const Ndarray& x, const int64_t* y) {
  auto scores = forward(x);
  auto dscores = scores.as_zeros();
  auto loss = SoftmaxLoss(scores, y, &dscores);
  auto dx = backward(dscores);
  // reg loss
  if (config_.reg > 0) {
    loss += config_.reg * 0.5 *
            ((conv_.w_ * conv_.w_).sum() + (affine_.w_ * affine_.w_).sum() +
             (affine2_.w_ * affine2_.w_).sum());
    conv_.dw_ += conv_.w_ * config_.reg;
    affine_.dw_ += affine_.w_ * config_.reg;
    affine2_.dw_ += affine2_.w_ * config_.reg;
  }
  return loss;
}

void SimpleConvNet::train(const Ndarray& x, std::vector<int64_t>& y,
                          const Ndarray& x_val, std::vector<int64_t>& y_val,
                          int epochs, int64_t batch, double lr,
                          int64_t eval_every) {
  assert(x.ndim() == 4);
  assert(x.shape(0) == y.size());
  assert(x_val.ndim() == 4);
  assert(x_val.shape(0) == y_val.size());
  int64_t N = y.size();
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int64_t i = 0; i < N; i += batch) {
      auto x_batch = x.slice(i, std::min(batch, N - i));
      const int64_t* y_batch = &y[i];
      double batchloss = loss(x_batch, y_batch);
#define SGD(layer, param) \
  layer.d##param *= lr;   \
  layer.param -= layer.d##param
      SGD(conv_, w_);
      SGD(conv_, b_);
      SGD(affine_, w_);
      SGD(affine_, b_);
      SGD(affine2_, w_);
      SGD(affine2_, b_);
#undef SGD
      losses_.push_back(batchloss);
      iter_++;
      if (iter_ % eval_every == 0) {
        std::cout << "iter:" << iter_ << " epoch:" << epoch
                  << " loss:" << batchloss << std::endl;
      }
    }
  }
}
