#include "cnn.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "layers.h"
#include "loss.h"
#include "ndarray.h"

namespace litecnn {

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
  assert(alpha > 0);
  return *this;
}

SimpleConvNet::SimpleConvNet(SimpleConvNet::Config config)
    : config_(config.validated()),
      conv_(config.filter_size, config.filter_size, config.input_depth,
            config.n_filters, 1, (config.filter_size - 1) / 2,
            config.weight_scale),
      pool_(2, 2, 2),
      affine_(config.n_filters * ((config.input_height + 1) / 2) *
                  ((config.input_width + 1) / 2),
              config.hidden_dim, config.weight_scale),
      affine2_(config.hidden_dim, config.n_classes, config.weight_scale),
      iter_(new std::atomic_int(0)) {}

Ndarray SimpleConvNet::forward(const Ndarray& x) {
  assert(x.ndim() == 4);
  assert(x.shape(1) == config_.input_depth);
  assert(x.shape(2) == config_.input_height);
  assert(x.shape(3) == config_.input_width);

  auto out1 = conv_.forward(x);
  auto out2 = relu_.forward(out1);
  auto out3 = pool_.forward(out2);
  shape_before_affine_ = out3.shape();
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
  auto dout3 = dout4.reshape(shape_before_affine_);
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
  return loss;
}

void SimpleConvNet::train(const Ndarray& x, const int64_t* y,
                          const Ndarray& x_val, const int64_t* y_val,
                          int epochs, int64_t batch, double lr,
                          int64_t log_every, int64_t eval_every) {
  assert(x.ndim() == 4);
  assert(x_val.ndim() == 4);
  int64_t N = x.shape(0);
  double batchloss = .0;
  for (int ep = 0; ep < epochs; ep++) {
    for (int64_t i = 0; i < N; i += batch) {
      auto N_batch = std::min(batch, N - i);
      auto x_batch = x.slice(i, N_batch);
      const int64_t* y_batch = y + i;
      SimpleConvNet snap = snapshot();
      batchloss = snap.loss(x_batch, y_batch);
#define FTRL(layer, param)                                                 \
  do {                                                                     \
    std::lock_guard<std::mutex> guard(*layer.lock_);                       \
    auto& n = layer.n##param;                                              \
    auto& z = layer.z##param;                                              \
    auto& g = snap.layer.d##param;                                         \
    auto& w = layer.param;                                                 \
    if (n.ndim() == 0) {                                                   \
      n = g.as_zeros();                                                    \
    }                                                                      \
    if (z.ndim() == 0) {                                                   \
      z = g.as_zeros();                                                    \
    }                                                                      \
    for (int64_t i0 = 0; i0 < w.shape(0); i0++) {                          \
      for (int64_t i1 = 0; i1 < w.shape(1); i1++) {                        \
        for (int64_t i2 = 0; i2 < w.shape(2); i2++) {                      \
          for (int64_t i3 = 0; i3 < w.shape(3); i3++) {                    \
            double gi = g.at(i0, i1, i2, i3);                              \
            double& ni = n.at(i0, i1, i2, i3);                             \
            double& zi = z.at(i0, i1, i2, i3);                             \
            double& wi = w.at(i0, i1, i2, i3);                             \
            double sigma =                                                 \
                (std::sqrt(ni + gi * gi) - std::sqrt(ni)) / config_.alpha; \
            zi += gi - sigma * wi;                                         \
            ni += gi * gi;                                                 \
            if (zi > config_.lambda1 || zi < -config_.lambda1) {           \
              wi = -(zi - (zi >= 0 ? 1 : -1) * config_.lambda1) /          \
                   ((config_.beta + std::sqrt(ni)) / config_.alpha +       \
                    config_.lambda2);                                      \
            } else {                                                       \
              wi = 0;                                                      \
            }                                                              \
            std::cout << gi << " " << ni << " " << zi << " " << wi         \
                      << std::endl;                                        \
          }                                                                \
        }                                                                  \
      }                                                                    \
    }                                                                      \
  } while (0)
      FTRL(conv_, w_);
      FTRL(conv_, b_);
      FTRL(affine_, w_);
      FTRL(affine_, b_);
      FTRL(affine2_, w_);
      FTRL(affine2_, b_);
#undef FTRL
      int curr = iter_->fetch_add(1) + 1;
      if (log_every > 0 && curr % log_every == 0) {
        std::cout << "iter:" << curr << " epoch:" << ep + 1
                  << " loss:" << batchloss << std::endl;
      }
      if (eval_every > 0 && curr % eval_every == 0) {
        double val_accuracy = eval(x_val, y_val);
        std::cout << "val_accuracy:" << val_accuracy << std::endl;
      }
    }
  }
  double val_accuracy = eval(x_val, y_val);
  std::cout << "final val accuracy:" << val_accuracy << " loss:" << batchloss
            << std::endl;
}  // namespace litecnn

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
  std::unique_ptr<int64_t[]> ypred(new int64_t[size]);
  predict(x, ypred.get());
  double match = 0.0;
  for (int64_t i = 0; i < size; i++) {
    if (y[i] == ypred[i]) {
      match += 1.0;
    }
  }
  return match / size;
}

SimpleConvNet SimpleConvNet::snapshot() {
  SimpleConvNet snap = *this;
  snap.conv_.w_ = snap.conv_.w_.fork();
  snap.conv_.b_ = snap.conv_.b_.fork();
  snap.affine_.w_ = snap.affine_.w_.fork();
  snap.affine_.b_ = snap.affine_.b_.fork();
  snap.affine2_.w_ = snap.affine2_.w_.fork();
  snap.affine2_.b_ = snap.affine2_.b_.fork();
  return snap;
}

}  // namespace litecnn
