from keras.engine.topology import Layer

from .. import activations


class GELU(Layer):

    def __init__(self, approximation, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.approximation = approximation

    # TODO: add learnable mu and sigma besides approximations of N(0, 1) cdf
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