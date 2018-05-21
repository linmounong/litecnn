#include <cassert>
#include <iostream>

#include "layers.h"
#include "matrix.h"
#include "ndarray.h"

void test_ndarray() {
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

  Ndarray a({2, 3},
            {
                1, 2, 3,  //
                4, 5, 6,  //
            });
  Ndarray b({3, 2},
            {
                7, 8,    //
                9, 10,   //
                11, 12,  //
            });
  Ndarray c({2, 2},
            {
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

  auto z = Ndarray::zeros_like(a);
  assert(z == Ndarray({2, 1, 3}, {0, 0, 0, 0, 0, 0}));
}

void test_matrix() {
  Matrix m1(3, 6);
  assert(m1.rows() == 3);
  assert(m1.cols() == 6);
  m1.zero();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      assert(m1.at(i, j) == 0);
    }
  }
  assert(m1.sum() == 0);

  m1.uniform(10);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      assert(m1.at(i, j) != 0);
      assert(m1.at(i, j) < 10);
      assert(m1.at(i, j) > -10);
    }
  }
  assert(m1.sum() != 0);

  Matrix m3 = m1.inner(m1);
  assert(m3.sum() > 0);

  // (np.arange(6).reshape(2, 3) + 1).dot(np.arange(6).reshape(3, 2) + 7)
  Matrix m4({{1, 2, 3}, {4, 5, 6}});
  Matrix m5({{7, 8}, {9, 10}, {11, 12}});
  assert(m4.dot(m5) == Matrix({{58, 64}, {139, 154}}));

  Matrix m6 = m5.T();
  assert(m6.rows() == 2);
  assert(m6.cols() == 3);
  assert(m6 == Matrix({{7, 9, 11}, {8, 10, 12}}));
  assert((m4 + m6) == Matrix({{8, 11, 14}, {12, 15, 18}}));

  Vector v(3);
  assert(v.size() == 3);
  v.zero();
  for (int i = 0; i < 3; i++) {
    assert(v[i] == 0);
  }
  for (int i = 0; i < 3; i++) {
    v[i] = i * i;
  }
  assert(v == Vector({0, 1, 4}));

  Matrix m({{1, 2, 3}, {4, 5, 6}});
  assert((m + v) == Matrix({{1, 3, 7}, {4, 6, 10}}));

  v.add(m, 0);
  assert(v == Vector({1, 3, 7}));
}

void test_layers() {
  Affine affine(3, 4);
  Matrix x(2, 3);
  x.uniform(1);
  auto out = affine.forward(x);
  assert(out.rows() == 2);
  assert(out.cols() == 4);
  auto dout = out;
  auto dx = affine.backward(dout);
  assert(dx.rows() == 2);
  assert(dx.cols() == 3);

  Relu relu;
  x = Matrix({{-1, 1}, {0, 2}});
  out = relu.forward(x);
  assert(out == Matrix({{0, 1}, {0, 2}}));
  dout = Matrix({{5, 6}, {7, 8}});
  dx = relu.backward(dout);
  assert(dx == Matrix({{0, 6}, {0, 8}}));

  MaxPool max_pool(2, 2, 2);
  x = Matrix({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  out = max_pool.forward(x);
  assert(out == Matrix({{5, 6}, {8, 9}}));
  dx = max_pool.backward(out);
  assert(dx == Matrix({{0, 0, 0}, {0, 5, 6}, {0, 8, 9}}));
}

int main() {
  test_ndarray();
  test_matrix();
  test_layers();
  std::cout << "all passed" << std::endl;
}
