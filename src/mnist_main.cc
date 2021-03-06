#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "cnn.h"
#include "mnist/mnist_reader.hpp"

const int kDefaultThreads = 4;

void ReadData(const std::string& path, litecnn::Ndarray* x,
              std::vector<int64_t>* y, litecnn::Ndarray* x_test,
              std::vector<int64_t>* y_test) {
  auto dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("mnist");
  assert(dataset.training_images.size() == dataset.training_labels.size());
  assert(dataset.test_images.size() == dataset.test_labels.size());
#define LOAD(src_x, src_y, target_x, target_y)                              \
  do {                                                                      \
    std::vector<int64_t> shuf(src_x.size());                                \
    std::iota(shuf.begin(), shuf.end(), 0);                                 \
    std::shuffle(shuf.begin(), shuf.end(), std::default_random_engine(42)); \
    *target_x = litecnn::Ndarray(src_x.size(), 1, 28, 28);                  \
    target_y->resize(src_x.size());                                         \
    for (int i = 0; i < src_x.size(); i++) {                                \
      assert(src_x[i].size() == 28 * 28);                                   \
      for (int j = 0; j < src_x[i].size(); j++) {                           \
        target_x->at(shuf[i], 0, j / 28, j % 28) = src_x[i][j];             \
      }                                                                     \
      (*target_y)[shuf[i]] = src_y[i];                                      \
    }                                                                       \
    std::cout << "loaded " << src_x.size() << " from " #src_x << std::endl; \
  } while (0)
  LOAD(dataset.training_images, dataset.training_labels, x, y);
  LOAD(dataset.test_images, dataset.test_labels, x_test, y_test);
#undef LOAD
}

int main(int argc, char* argv[]) {
  int n_threads = kDefaultThreads;
  if (argc == 2) {
    n_threads = std::atoi(argv[1]);
  }
  std::cout << "training using " << n_threads << " threads" << std::endl;

  litecnn::Ndarray x;
  litecnn::Ndarray x_test;
  std::vector<int64_t> y;
  std::vector<int64_t> y_test;
  ReadData("mnist", &x, &y, &x_test, &y_test);

  litecnn::SimpleConvNet::Config config;
  config.input_height = 28;
  config.input_width = 28;
  config.input_depth = 1;
  config.n_filters = 10;
  config.filter_size = 5;
  config.hidden_dim = 50;
  config.weight_scale = 1e-2;
  config.n_classes = 10;
  config.reg = 0.5;
  litecnn::SimpleConvNet cnn(config);
  auto start = std::chrono::steady_clock::now();

  std::cout << "warming up..." << std::endl;
  auto warm_up = [&cnn, &x, &y, n_threads](int i) {
    int train_i = x.shape(0) / n_threads * i;
    int train_n = std::min(100LL, x.shape(0) - train_i);
    cnn.train(x.slice(train_i, train_n), &y[train_i],  // train data
              x, &y[0],  // eval data, doesn't matter here
              1,         // epochs
              100,       // batch
              0.01,      // lr
              1,         // log_every
              0);        // eval_every
  };
  for (int i = 0; i < n_threads; ++i) {
    warm_up(i);
  }
  for (int i = n_threads - 1; i >= 0; --i) {
    warm_up(i);
  }

  auto thread_func = [&cnn, &x, &y, &x_test, &y_test, n_threads](int i) {
    int train_i = x.shape(0) / n_threads * i;
    int train_n = std::min(x.shape(0) / n_threads, x.shape(0) - train_i);
    // int test_i = x_test.shape(0) / n_threads * i;
    // int test_n = std::min(x_test.shape(0) / n_threads, x.shape(0) - test_i);
    int test_i = 0;
    int test_n = 1000;
    std::cout << "thread " << i << "(" << std::this_thread::get_id()
              << ") starting... from " << train_i << " count " << train_n
              << std::endl;
    cnn.train(x.slice(train_i, train_n), &y[train_i],         // train data
              x_test.slice(test_i, test_n), &y_test[test_i],  // eval data
              2,                                              // epochs
              100,                                            // batch
              0.005,                                          // lr
              10,                                             // log_every
              100);                                           // eval_every
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < n_threads; ++i) {
    threads.emplace_back(thread_func, i);
  }
  for (auto& t : threads) {
    t.join();
  }
  auto end = std::chrono::steady_clock::now();
  std::cout
      << "training took "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << "s\n";
  std::cout << "final test accuracy " << cnn.eval(x_test, &y_test[0])
            << std::endl;
}
