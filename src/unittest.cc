#include <cassert>
#include <cmath>
#include <iostream>

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
  std::vector<float> expected{
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

  auto f = r.fork();
  f.at(0, 2, 0, 1) = 60;  // forked data
  assert(expected == data);

  auto m2 = m * 2;
  auto& data2 = *m2.data();
  for (int i = 0; i < expected.size(); i++) {
    assert(data2[i] == data[i] * 2);
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

  a = a.reshape(2, 1, 3);
  b = b.reshape(1, 2, 3);
  c = Ndarray({2, 2, 3}, {8, 10, 12, 11, 13, 15, 11, 13, 15, 14, 16, 18});
  assert(a + b == c);
  b = b.reshape(2, 3);
  assert(a + b == c);

  auto z = a.as_zeros();
  assert(z == Ndarray({2, 1, 3}, {0, 0, 0, 0, 0, 0}));
}

void TestLayers() {
  Affine affine(3, 4);
  Ndarray x(2, 3);
  x.uniform(1);
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
            1,   // s
            1);  // p
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
  dout.uniform(1);
  dx = conv.backward(dout);
  assert(dx.shape() == x.shape());
}

void TestLoss() {
  Ndarray x({2, 3}, {1, 2, 3, 4, 5, 6});
  std::vector<int64_t> y = {1, 2};
  Ndarray dx = x.as_zeros();
  float loss = SoftmaxLoss(x, y, &dx);
  assert(std::abs(loss - 0.9076059644443804) < 1e-6);
  Ndarray expected({2, 3}, {0.04501529, -0.37763576, 0.33262048, 0.04501529,
                            0.12236424, -0.16737952});
  assert(std::abs(dx.sum() - expected.sum()) < 1e-6);
}

int main() {
  TestNdarray();
  TestLayers();
  TestLoss();
  std::cout << "all passed" << std::endl;
}
