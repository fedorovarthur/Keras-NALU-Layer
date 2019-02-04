from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import log

import keras.backend as K


def relational_loss(y_true, y_pred, loss, alpha=.2, threshold=0):
    assert K.shape(y_true) == K.shape(y_pred), 'Shapes are inconsistent'

    def _r(x):
        return K.relu(K.dot(x, K.transpose(x)), threshold=threshold)

    return (1 - alpha) * loss(y_true, y_pred) + alpha * loss(_r(y_true), _r(y_pred))


def vae_loss(y_true, y_pred, reconstruction_loss, mean, log_var):
    assert K.shape(y_true) == K.shape(y_pred), 'Shapes are inconsistent'

    shape = K.shape(y_true)
    log_shape_sum = sum((log(shape[i]) for i in range(1, len(shape))))
    reconstruction_log_loss = log_shape_sum + K.log(reconstruction_loss(y_true, y_pred))
    kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.mean(reconstruction_log_loss + kl_loss)
