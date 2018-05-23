#include "cnn.h"

#include <cassert>
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

void SimpleConvNet::train(const Ndarray& x, const int64_t* y,
                          const Ndarray& x_val, const int64_t* y_val,
                          int epochs, int64_t batch, double lr,
                          int64_t log_every, int64_t eval_every) {
  assert(x.ndim() == 4);
  assert(x_val.ndim() == 4);
  int64_t N = x.shape(0);
  for (int ep = 0; ep < epochs; ep++) {
    for (int64_t i = 0; i < N; i += batch) {
      auto N_batch = std::min(batch, N - i);
      auto x_batch = x.slice(i, N_batch);
      const int64_t* y_batch = y + i;
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
      if (log_every > 0 && iter_ % log_every == 0) {
        std::cout << "iter:" << iter_ << " epoch:" << ep + 1
                  << " loss:" << batchloss << std::endl;
      }
      if (eval_every > 0 && iter_ % eval_every == 0) {
        double val_accuracy = eval(x_val, y_val);
        std::cout << "val_accuracy:" << val_accuracy << std::endl;
      }
    }
  }
  double train_accuracy = eval(x, y);
  double val_accuracy = eval(x_val, y_val);
  std::cout << "final train_accuracy:" << train_accuracy
            << " val_accuracy:" << val_accuracy << " loss:" << *losses_.rbegin()
            << std::endl;
}

void SimpleConvNet::predict(const Ndarray& x, int64_t* y) {
  auto scores = forward(x);
  assert(scores.ndim() == 2);
  int64_t size = scores.shape(0);
  int64_t classes = scores.shape(1);
  for (int64_t i = 0; i < size; i++) {
    int64_t argmax = 0;
    double max = scores.at(i, 0);
    for (int64_t j = 1; j < classes; j++) {
      if (max < scores.at(i, j)) {
        max = scores.at(i, j);
        argmax = j;
      }
    }
    y[i] = argmax;
  }
}

double SimpleConvNet::eval(const Ndarray& x, const int64_t* y) {
  int64_t size = x.shape(0);
  int64_t ypred[size];
  predict(x, ypred);
  double match = 0.0;
  for (int64_t i = 0; i < size; i++) {
    if (y[i] == ypred[i]) {
      match += 1.0;
    }
  }
  return match / size;
}
