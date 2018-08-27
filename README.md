My implementation of Convolutional Neural Networks for practice purposes.

```bash
# build
make

# unit test
make test

# train a cnn over mnist data
make train
```

Training using 4 threads took 826s on my macbook with a test accuracy of 96.11%.
Meanwhile training using 1 thread took 1582s with a test accuracy of 96.15%.

Checkout `trainlog/` for the latest training log.
