from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import pi, sqrt

import keras.backend as K


def gelu(x, approximation='tanh'):
    assert approximation in ('sigmoid', 'tanh'), \
        'Approximation method must be chosen from [tanh, sigmoid]'

    if approximation == 'tanh':
        return .5 * x * (1 + K.tanh(sqrt(2/pi) * (x + .044715 * x ** 3)))
    else:
        return x * K.sigmoid(1.702 * x)


def silu(x):
    return x * K.sigmoid(x)


def swish(x, beta):
    return x * K.sigmoid(beta * x)


def nac(x, w, m):
    return K.dot(x, K.tanh(w) * K.sigmoid(m))


def log_nac(x, w, m):
    return K.exp(K.dot(K.log(K.abs(x) + K.epsilon()), nac(x, w, m)))
