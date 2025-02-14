import numpy as np
import tensorflow as tf

class Normal:
    def __init__(self, mu, sigma, name=None):
        # Create variables if raw values provided, otherwise use existing variables
        self.mu = tf.Variable(mu) if not isinstance(mu, tf.Variable) else mu
        self.sigma = tf.Variable(sigma) if not isinstance(sigma, tf.Variable) else sigma
        self.name = name

    def kl_divergence(self, other):
        """Calculate KL divergence between this normal and another normal distribution"""
        return (
            (tf.square(self.sigma) + tf.square(self.mu - other.mu)) / (2 * tf.square(other.sigma))
            - 0.5
            + tf.math.log(other.sigma / self.sigma)
        )
            

    def pdf(self, x):
        """Calculate probability density at point x"""
        return (
            1 / (self.sigma * tf.sqrt(2 * tf.constant(np.pi)))
            * tf.exp(-0.5 * tf.square((x - self.mu) / self.sigma))
        )

    @property
    def trainable_variables(self):
        """Return list of trainable variables for optimization"""
        return [self.mu, self.sigma]

class Linear:
    def __init__(self, b0, b1, name=None):
        # Create variables if raw values provided, otherwise use existing variables
        self.b0 = tf.Variable(b0) if not isinstance(b0, tf.Variable) else b0
        self.b1 = tf.Variable(b1) if not isinstance(b1, tf.Variable) else b1
        self.name = name

    def function(self, x):
        return self.b0 + self.b1 * x

    @property
    def trainable_variables(self):
        """Return list of trainable variables for optimization"""
        return [self.b0, self.b1]
