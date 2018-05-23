#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>

#include "cnn.h"
#include "layers.h"
#include "loss.h"
#include "ndarray.h"

void TestNdarray() {
  Ndarray m(3, 6);
  auto& data = *m.data();
  assert(m.ndim() == 2);
  assert(m.shape(0) == 3);
  assert(m.shape(1) == 6);
  assert((std::vector<int64_t>{3, 6} == m.shape()));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      assert(m.at(i, j) == 0);
      m.at(i, j) = i + j;
    }
  }
  std::vector<double> expected{
      0, 1, 2, 3, 4, 5,  //
      1, 2, 3, 4, 5, 6,  //
      2, 3, 4, 5, 6, 7,  //
  };
  assert(expected == data);
  assert(63 == m.sum());
  assert(std::vector<int64_t>{6} == m.sum(0).shape());
  assert(std::vector<int64_t>{3} == m.sum(1).shape());

  auto t = m.T();
  assert(t.ndim() == 2);
  assert(t.shape(0) == 6);
  assert(t.shape(1) == 3);
  assert(expected == data);  // underlying data not changed

  expected[3] = 20;
  t.at(3, 0) = 20;
  assert(expected == data);

  auto r = m.reshape(3, 3, 1, 2);
  expected[5] = 50;
  r.at(0, 2, 0, 1) = 50;
  assert(expected == data);
  auto r2 = m.reshape(-1, 3, 1, 2);
  assert(r == r2);

  auto f = r.fork();
  f.at(0, 2, 0, 1) = 60;  // forked data
  assert(expected == data);

  auto m2 = m * 2;
  auto& data2 = *m2.data();
  for (int i = 0; i < data.size(); i++) {
    assert(data2[i] == data[i] * 2);
  }
  auto m3 = m - 3;
  auto& data3 = *m3.data();
  for (int i = 0; i < data.size(); i++) {
    assert(data3[i] == data[i] - 3);
  }
  auto m4 = m * m;
  auto& data4 = *m4.data();
  for (int i = 0; i < data.size(); i++) {
    assert(data4[i] == data[i] * data[i]);
  }

  Ndarray a({2, 3}, {
                        1, 2, 3,  //
                        4, 5, 6,  //
                    });
  Ndarray b({3, 2}, {
                        7, 8,    //
                        9, 10,   //
                        11, 12,  //
                    });
  Ndarray c({2, 2}, {
                        58, 64,    //
                        139, 154,  //
                    });
  assert(a.dot(b) == c);

  // >>> a = (np.arange(6)+1).reshape(2, 1, 3)
  // >>> b = (np.arange(6)+7).reshape(1, 2, 3)
  // >>> (a+b).shape
  // (2, 2, 3)
  // >>> a+b
  // array([[[ 8, 10, 12],
  //         [11, 13, 15]],
  //
  //        [[11, 13, 15],
  //         [14, 16, 18]]])
  // >>> a-b
  // array([[[-6, -6, -6],
  //         [-9, -9, -9]],
  //
  //        [[-3, -3, -3],
  //         [-6, -6, -6]]])

  a = a.reshape(2, 1, 3);
  b = b.reshape(1, 2, 3);
  c = Ndarray({2, 2, 3}, {8, 10, 12, 11, 13, 15, 11, 13, 15, 14, 16, 18});
  assert(a + b == c);
  b = b.reshape(2, 3);
  assert(a + b == c);
  auto d = Ndarray({2, 2, 3}, {-6, -6, -6, -9, -9, -9, -3, -3, -3, -6, -6, -6});
  assert(a - b == d);

  auto z = a.as_zeros();
  assert(z == Ndarray({2, 1, 3}, {0, 0, 0, 0, 0, 0}));

  auto s = a.reshape(3, 2).slice(1, 2);
  assert(s == Ndarray({2, 2}, {3, 4, 5, 6}));
}

void TestLayers() {
  Affine affine(3, 4, 1e-3);
  Ndarray x(2, 3);
  x.gaussian(1);
  auto out = affine.forward(x);
  assert(out.shape(0) == 2);
  assert(out.shape(1) == 4);
  auto dout = out;
  auto dx = affine.backward(dout);
  assert(dx.shape(0) == 2);
  assert(dx.shape(1) == 3);

  Relu relu;
  x = Ndarray({2, 2}, {-1, 1, 0, 2});
  out = relu.forward(x);
  assert(out == Ndarray({2, 2}, {0, 1, 0, 2}));
  dout = Ndarray({2, 2}, {5, 6, 7, 8});
  dx = relu.backward(dout);
  assert(dx == Ndarray({2, 2}, {0, 6, 0, 8}));

  MaxPool pool1(2, 2, 2);
  x = Ndarray({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  out = pool1.forward(x);
  assert(out == Ndarray({2, 2}, {5, 6, 8, 9}));
  dx = pool1.backward(out);
  assert(dx == Ndarray({3, 3}, {0, 0, 0, 0, 5, 6, 0, 8, 9}));

  MaxPool poo2(2, 2, 2);
  x = Ndarray({2, 3, 3},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  out = poo2.forward(x);
  assert(out == Ndarray({2, 2, 2}, {5, 6, 8, 9, 5, 6, 8, 9}));
  dx = poo2.backward(out);
  assert(dx == Ndarray({2, 3, 3},
                       {0, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0, 5, 6, 0, 8, 9}));

  Conv conv(3,   // fh
            3,   // fw
            1,   // fc
            6,   // fn
            1,   // stride
            1,   // pad
            1);  // scale
  x = Ndarray(
      {2, 1, 4, 5},
      {
          0.11539938, 0.49015755, 0.24796755, 0.18262233, 0.8828635,  // #1
          0.22141799, 0.07918406, 0.76438275, 0.77657192, 0.01950792,
          0.9445943,  0.15560637, 0.09516008, 0.25185071, 0.24268301,
          0.05987559, 0.90623107, 0.61799713, 0.77170657, 0.90150478,
          0.77493478, 0.1367159,  0.09158035, 0.94439507, 0.72882132,  // #2
          0.98975397, 0.4502993,  0.63584008, 0.48091516, 0.87262581,
          0.86261557, 0.16982389, 0.41723116, 0.18291976, 0.32968293,
          0.51262869, 0.82402136, 0.1498909,  0.4061689,  0.24924964,
      });
  out = conv.forward(x);
  assert((std::vector<int64_t>{2, 6, 4, 5} == out.shape()));
  dout = out.as_zeros();
  dout.gaussian(1);
  dx = conv.backward(dout);
  assert(dx.shape() == x.shape());
}

void TestLoss() {
  Ndarray x({2, 3}, {1, 2, 3, 4, 5, 6});
  int64_t y[2] = {1, 2};
  Ndarray dx = x.as_zeros();
  double loss = SoftmaxLoss(x, y, &dx);
  assert(std::abs(loss - 0.9076059644443804) < 1e-6);
  Ndarray expected({2, 3}, {0.04501529, -0.37763576, 0.33262048, 0.04501529,
                            0.12236424, -0.16737952});
  assert(std::abs(dx.sum() - expected.sum()) < 1e-6);
}

Ndarray NumericGrad(std::function<double()> func, Ndarray x, double h) {
  auto dx = x.as_zeros();
  for (int64_t i0 = 0; i0 < x.shape(0); i0++) {
    for (int64_t i1 = 0; i1 < x.shape(1); i1++) {
      for (int64_t i2 = 0; i2 < x.shape(2); i2++) {
        for (int64_t i3 = 0; i3 < x.shape(3); i3++) {
          x.at(i0, i1, i2, i3) += h;
          auto s1 = func();
          x.at(i0, i1, i2, i3) -= h + h;
          auto s2 = func();
          dx.at(i0, i1, i2, i3) = (s1 - s2) / (h + h);
          x.at(i0, i1, i2, i3) += h;
        }
      }
    }
  }
  return dx;
}

void TestCnn() {
  {
    SimpleConvNet::Config config;
    config.input_height = 32;
    config.input_width = 32;
    config.input_depth = 3;
    config.n_filters = 32;
    config.filter_size = 7;
    config.hidden_dim = 100;
    config.weight_scale = 1e-3;
    config.n_classes = 10;
    config.reg = 0;

    Ndarray x({5, config.input_depth, config.input_height, config.input_width},
              nullptr);
    x.gaussian(1);
    int64_t y[] = {1, 2, 3, 4, 5};

    auto loss = SimpleConvNet(config).loss(x, y);
    assert(std::abs(loss - (-std::log(0.1))) < 1e-3);

    config.reg = 0.5;
    auto loss2 = SimpleConvNet(config).loss(x, y);
    assert(loss2 > loss);
    assert(loss2 < loss + 1);
  }
  {
    SimpleConvNet::Config config;
    config.input_height = 16;
    config.input_width = 16;
    config.input_depth = 3;
    config.n_filters = 3;
    config.filter_size = 3;
    config.hidden_dim = 7;
    config.weight_scale = 1e-2;
    config.n_classes = 10;
    config.reg = 0;
    SimpleConvNet cnn(config);

    Ndarray x({2, config.input_depth, config.input_height, config.input_width},
              nullptr);
    x.gaussian(1);
    int64_t y[2] = {1, 2};
    auto loss = cnn.loss(x, y);

    Ndarray dscores({x.shape(0), config.n_classes}, nullptr);
#define TestGrad(target, grad)                                      \
  do {                                                              \
    auto grad2 = NumericGrad(                                       \
        [&x, y, &cnn, &dscores]() {                                 \
          return SoftmaxLoss(cnn.forward(x), y, &dscores);          \
        },                                                          \
        target, 1e-4);                                              \
    assert(grad.shape() == grad2.shape());                          \
    auto diff = grad - grad2;                                       \
    std::cout << "max diff " #grad ": " << diff.max() << std::endl; \
  } while (0)
    TestGrad(cnn.conv_.w_, cnn.conv_.dw_);
    TestGrad(cnn.conv_.b_, cnn.conv_.db_);
    TestGrad(cnn.affine_.w_, cnn.affine_.dw_);
    TestGrad(cnn.affine_.b_, cnn.affine_.db_);
    TestGrad(cnn.affine2_.w_, cnn.affine2_.dw_);
    TestGrad(cnn.affine2_.b_, cnn.affine2_.db_);
#undef TestGrad
  }
  {
    SimpleConvNet::Config config;
    config.input_height = 32;
    config.input_width = 32;
    config.input_depth = 3;
    config.n_filters = 7;
    config.filter_size = 3;
    config.hidden_dim = 20;
    config.weight_scale = 1e-3;
    config.n_classes = 10;
    config.reg = 0.1;
    SimpleConvNet cnn(config);

    Ndarray x(
        {100, config.input_depth, config.input_height, config.input_width},
        nullptr);
    x.gaussian(1);
    std::vector<int64_t> y(100);
    cnn.train(x, y,  // train data
              x, y,  // eval data
              10,    // epochs
              40,    // batch
              0.01,  // lr
              2);    // every_every

    std::cout << "check if the model overfits" << std::endl;
  }
}

int main() {
  TestNdarray();
  TestLayers();
  TestLoss();
  TestCnn();
  std::cout << "all passed" << std::endl;
}
