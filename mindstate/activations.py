import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def swish(x):
    return sigmoid(x) * x
