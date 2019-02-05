from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K

from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints

from ..activations import nac, log_nac, gelu, swish
from .advanced_activations import GELU, Swish


class ParametricGELU(GELU):

    def __init__(self, approximation='tanh',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParametricGELU, self).__init__(**kwargs)
        self.approximation = approximation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0]

        self.mu = self.add_weight(shape=(input_dim,),
                                  initializer=self.kernel_initializer,
                                  name='mu',
                                  regularizer = self.kernel_regularizer,
                                  constraint = self.kernel_constraint)
        self.sigma = self.add_weight(shape=(input_dim,),
                                     initializer=self.kernel_initializer,
                                     name='sigma',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return self.mu + self.sigma * gelu(inputs, approximation=self.approximation)


class ParametricSwish(Swish):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParametricSwish, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0]

        self.beta = self.add_weight(shape=(input_dim,),
                                  initializer=self.kernel_initializer,
                                  name='beta',
                                  regularizer = self.kernel_regularizer,
                                  constraint = self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return swish(inputs, self.beta)


class NAC(Layer):

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NAC, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='M_hat',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_hat',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return nac(inputs, self.M_hat, self.W_hat)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(NAC, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class NALU(Layer):

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape)
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='M_hat',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_hat',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        self.G = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='G',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        g = K.sigmoid(K.dot(inputs, self.G))
        return g * nac(inputs, self.W_hat, self.M_hat) + (1 - g) * log_nac(inputs, self.W_hat, self.M_hat)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(NALU, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))
