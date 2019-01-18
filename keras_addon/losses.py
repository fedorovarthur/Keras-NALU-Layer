from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K


def relational_loss(y_true, y_pred, loss, alpha=.5, threshold=0):

    def _r(x):
        return K.relu(K.dot(x, x.T), threshold=threshold)

    return (1 - alpha) * loss(y_true, y_pred) + alpha * loss(_r(y_true), _r(y_pred))
