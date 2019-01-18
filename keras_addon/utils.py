from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K


def reparametrization_trick(sample, mu, sigma):
    return mu + K.exp(.5 * sigma) * sample


def sampling_with_reparametrization(mean, log_var):
    batch, dim = K.shape(mean)[0], K.int_shape(mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return reparametrization_trick(epsilon, mean, log_var)
