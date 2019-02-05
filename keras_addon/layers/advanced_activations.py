from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.topology import Layer

from .. import activations


class GELU(Layer):

    def __init__(self, approximation='tanh', **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.approximation = approximation

    def call(self, inputs, **kwargs):
        return activations.gelu(inputs, approximation=self.approximation)

    def get_config(self):
        config = {'approximation': self.approximation}
        base_config = super(GELU, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class SiLU(Layer):

    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return activations.silu(inputs)

    def get_config(self):
        base_config = super(SiLU, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, beta=1, **kwargs):
        return activations.swish(inputs, beta)

    def get_config(self):
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
