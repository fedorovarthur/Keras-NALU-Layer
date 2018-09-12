import keras.backend as K
from keras.engine.topology import Layer

class NAC(Layer):

    def __init__(self, output_dim, kernel_initializer, **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        super(NAC, self).__init__(**kwargs)

    def build(self, input_shape):
        self._M_hat = self.add_weight(name='M_hat', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self._W_hat = self.add_weight(name='W_hat', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(NAC, self).build(input_shape)

    def call(self, x):
        self._W = K.tanh(self._W_hat)*K.sigmoid(self._M_hat)
        return K.dot(x, self._W)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class NALU(Layer):

    def __init__(self, output_dim, epsilon = 10e-6, kernel_initializer = 'glorot_normal', **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        K.set_epsilon(epsilon)
        super(NALU, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W_hat = self.add_weight(name='W_hat', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self._M_hat = self.add_weight(name='M_hat', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self._G = self.add_weight(name='G', 
                                  shape=(input_shape[1], self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        super(NALU, self).build(input_shape)

    def call(self, x):
        self._W = K.tanh(self._W_hat)*K.sigmoid(self._M_hat)
        self._a = K.dot(x, self._W)
        self._m = K.exp(K.dot(K.log(K.abs(x) + K.epsilon()), self._W))
        self._g = K.dot(x, self._G)
        return self._g*self._a + (1 - self._g)*self._m

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
