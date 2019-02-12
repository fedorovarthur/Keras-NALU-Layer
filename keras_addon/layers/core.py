from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K

from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints

from ..activations import nac, log_nac, gelu, swish, lelu
from .advanced_activations import GELU, Swish, LELU


class ParametricGELU(GELU):

    def __init__(self, approximation='tanh',
                 mu_initializer='zeros',
                 mu_regularizer=None,
                 mu_constraint=None,
                 sigma_initializer='ones',
                 sigma_regularizer=None,
                 sigma_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParametricGELU, self).__init__(**kwargs)
        self.approximation = approximation
        self.mu_initializer = initializers.get(mu_initializer)
        self.mu_regularizer = regularizers.get(mu_regularizer)
        self.mu_constraint = constraints.get(mu_constraint)
        self.sigma_initializer = initializers.get(sigma_initializer)
        self.sigma_regularizer = regularizers.get(sigma_regularizer)
        self.sigma_constraint = constraints.get(sigma_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.mu = self.add_weight(shape=(input_dim,),
                                  initializer=self.mu_initializer,
                                  name='mu',
                                  regularizer = self.mu_regularizer,
                                  constraint = self.mu_constraint)
        self.sigma = self.add_weight(shape=(input_dim,),
                                     initializer=self.sigma_initializer,
                                     name='sigma',
                                     regularizer=self.sigma_regularizer,
                                     constraint=self.sigma_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return self.mu + K.exp(.5 * self.sigma) * gelu(inputs, approximation=self.approximation)

    def get_config(self):
        config = {
            'approximation': self.approximation,
            'mu_initializer': initializers.serialize(self.mu_initializer),
            'mu_regularizer': regularizers.serialize(self.mu_regularizer),
            'mu_constraint': constraints.serialize(self.mu_constraint),
            'sigma_initializer': initializers.serialize(self.sigma_initializer),
            'sigma_regularizer': regularizers.serialize(self.sigma_regularizer),
            'sigma_constraint': constraints.serialize(self.sigma_constraint)
        }
        base_config = super(ParametricGELU, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class ParametricSwish(Swish):

    def __init__(self,
                 beta_initializer='ones',
                 beta_regularizer=None,
                 beta_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParametricSwish, self).__init__(**kwargs)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.beta = self.add_weight(shape=(input_dim,),
                                  initializer=self.beta_initializer,
                                  name='beta',
                                  regularizer = self.beta_regularizer,
                                  constraint = self.beta_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return swish(inputs, self.beta)

    def get_config(self):
        config = {
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(ParametricSwish, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class ParametricLELU(LELU):

    def __init__(self,
                 mu_initializer='zeros',
                 mu_regularizer=None,
                 mu_constraint=None,
                 s_initializer='ones',
                 s_regularizer=None,
                 s_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ParametricLELU, self).__init__(**kwargs)
        self.mu_initializer = initializers.get(mu_initializer)
        self.mu_regularizer = regularizers.get(mu_regularizer)
        self.mu_constraint = constraints.get(mu_constraint)
        self.s_initializer = initializers.get(s_initializer)
        self.s_regularizer = regularizers.get(s_regularizer)
        self.s_constraint = constraints.get(s_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.mu = self.add_weight(shape=(input_dim,),
                                  initializer=self.mu_initializer,
                                  name='mu',
                                  regularizer = self.mu_regularizer,
                                  constraint = self.mu_constraint)
        self.s = self.add_weight(shape=(input_dim,),
                                 initializer=self.s_initializer,
                                 name='s',
                                 regularizer = self.s_regularizer,
                                 constraint = self.s_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return lelu(inputs, self.mu, self.s)

    def get_config(self):
        config = {
            'mu_initializer': initializers.serialize(self.mu_initializer),
            'mu_regularizer': regularizers.serialize(self.mu_regularizer),
            'mu_constraint': constraints.serialize(self.mu_constraint),
            's_initializer': initializers.serialize(self.s_initializer),
            's_regularizer': regularizers.serialize(self.s_regularizer),
            's_constraint': constraints.serialize(self.s_constraint)
        }
        base_config = super(ParametricLELU, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class NAC(Layer):

    def __init__(self, units,
                 M_initializer='glorot_uniform',
                 M_regularizer=None,
                 M_constraint=None,
                 W_initializer='glorot_uniform',
                 W_regularizer=None,
                 W_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NAC, self).__init__(**kwargs)
        self.units = units
        self.M_initializer = initializers.get(M_initializer)
        self.M_regularizer = regularizers.get(M_regularizer)
        self.M_constraint = constraints.get(M_constraint)
        self.W_initializer = initializers.get(W_initializer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.M = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.M_initializer,
                                     name='M',
                                     regularizer=self.M_regularizer,
                                     constraint=self.M_constraint)
        self.W = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.W_initializer,
                                     name='W',
                                     regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        return nac(inputs, self.M, self.W)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'M_initializer': initializers.serialize(self.M_initializer),
            'M_regularizer': regularizers.serialize(self.M_regularizer),
            'M_constraint': constraints.serialize(self.M_constraint),
            'W_initializer': initializers.serialize(self.W_initializer),
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint)
        }
        base_config = super(NAC, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class NALU(Layer):

    def __init__(self, units,
                 M_initializer='glorot_uniform',
                 M_regularizer=None,
                 M_constraint=None,
                 W_initializer='glorot_uniform',
                 W_regularizer=None,
                 W_constraint=None,
                 G_initializer='glorot_uniform',
                 G_regularizer=None,
                 G_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.M_initializer = initializers.get(M_initializer)
        self.M_regularizer = regularizers.get(M_regularizer)
        self.M_constraint = constraints.get(M_constraint)
        self.W_initializer = initializers.get(W_initializer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.G_initializer = initializers.get(G_initializer)
        self.G_regularizer = regularizers.get(G_regularizer)
        self.G_constraint = constraints.get(G_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape)
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.M = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.M_initializer,
                                 name='M',
                                 regularizer=self.M_regularizer,
                                 constraint=self.M_constraint)
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.W_initializer,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.G = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.G_initializer,
                                 name='G',
                                 regularizer=self.G_regularizer,
                                 constraint=self.G_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        g = K.sigmoid(K.dot(inputs, self.G))
        return g * nac(inputs, self.W, self.M) + (1 - g) * log_nac(inputs, self.W, self.M)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'M_initializer': initializers.serialize(self.M_initializer),
            'M_regularizer': regularizers.serialize(self.M_regularizer),
            'M_constraint': constraints.serialize(self.M_constraint),
            'W_initializer': initializers.serialize(self.W_initializer),
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'G_initializer': initializers.serialize(self.G_initializer),
            'G_regularizer': regularizers.serialize(self.G_regularizer),
            'G_constraint': constraints.serialize(self.G_constraint),
        }
        base_config = super(NALU, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))


class LayerNormalization(Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 beta_initializer='zeros',
                 beta_regularizer=None,
                 beta_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        input_dim = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(shape=input_dim,
                                         initializer=self.gamma_initializer,
                                         name='gamma',
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        if self.center:
            self.beta = self.add_weight(shape=input_dim,
                                        initializer=self.beta_initializer,
                                        name='beta',
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint)
        }
        base_config = super(LayerNormalization, self).get_config()
        full_config = config.update(base_config)
        return dict(list(full_config.items()))
