#include <cassert>
#include <iostream>

// #include "layers.h"
#include "matrix.h"
#include "vector.h"

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
  assert(m6 == Matrix({{7, 9, 11}, {8, 10, 12}}));
  assert((m4 + m6) == Matrix({{8, 11, 14}, {12, 15, 18}}));
}

// void test_layers() {
//   AffineLayer affine(3, 4);
//   Matrix x(2, 3);
//   x.uniform(1);
//   auto out = affine.forward(x);
//   assert(out.rows() == 2);
//   assert(out.cols() == 4);
// }

void test_vector() {
  Vector v(10);
  assert(v.size() == 10);
  v.zero();
  for (int i = 0; i < 10; i++) {
    assert(v[i] == 0);
  }
  for (int i = 0; i < 10; i++) {
    v[i] = i * i;
  }
  for (int i = 0; i < 10; i++) {
    assert(v[i] == i * i);
  }
}

int main() {
  test_matrix();
  test_vector();
  std::cout << "all passed" << std::endl;
}
