import keras
import numpy as np


def load_mnist_data():
    """
    Loads MNIST data using tf.keras.datasets.mnist.load_data().
    Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test
