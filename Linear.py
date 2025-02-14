# This file has been split into two files:
# - agent.py: Contains the Agent class implementation
# - run_simulation.py: Contains the code to run the simulation

# Please use those files instead of this one.

from normal import Normal, Linear
from World import World
import numpy as np
import tensorflow as tf

'''
Experiment to see if this generally works
Get better learning rates and optimizers
Decide what to do with the fact that I model it with easy normals (is it okay because ill switch to nns anyway?)
jump to quadratics

'''

class Agent:
    def __init__(self, world):
        # Create the probabilistic models with TF variables
        self.px = Normal(0.0, 1.0, name='px')
        self.qx = Normal(self.px.mu, self.px.sigma, name='qx')  # Initialize q(x) to match p(x)
        self.pyx = Linear(0.0, 5.0, name='pyx')
        self.pypx_sigma = tf.Variable(1.0, name='pypx_sigma')
        self.world = world
        
        # Initialize optimizers with appropriate learning rates
        self.qx_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0)
        self.pyx_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        self.px_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    def get_px(self, x):
        return self.px.pdf(x)

    def get_pyx(self, x):
        return Normal(self.pyx.function(x), self.pypx_sigma).pdf(x)

    def qx(self, x):
        return self.qx.pdf(x)

    def _get_complexity(self):
        return self.qx.kl_divergence(self.px)

    def _get_accuracy(self, y):
        mu_function = self.pyx.function(self.qx.mu)
        sigma_function = self.qx.sigma * self.pyx.b1

        return -0.5 * tf.math.log(2 * tf.constant(np.pi)) - tf.math.log(self.pypx_sigma) - \
               0.5 * (tf.square(y - mu_function) + tf.square(sigma_function)) / tf.square(self.pypx_sigma)
    
    def get_surprise(self, y=None):
        """
        Computes the expected surprise, i.e., the expected value of -ln p(y) under the generative model.

        Given that both p(y|x) and p(x) are normal distributions, we can calculate this analytically.
        """
        if y is None:
            y = self.world.observe()

        # The marginal p(y) is also a normal distribution with:
        # mean = self.pyx.function(self.px.mu)
        # variance = self.pypx_sigma^2 + (self.pyx.b1^2) * self.px.sigma^2
        mean_y = self.pyx.function(self.px.mu)
        variance_y = tf.square(self.pypx_sigma) + tf.square(self.pyx.b1) * tf.square(self.px.sigma)

        # Calculate the surprise as -ln p(y) where p(y) is a normal distribution
        return 0.5 * tf.math.log(2 * tf.constant(np.pi) * variance_y) + \
               0.5 * tf.square(y - mean_y) / variance_y
        
    
    def get_vfe(self, y=None): 
        if y is None:
            y = self.world.observe()

        complexity = self._get_complexity()
        accuracy = self._get_accuracy(y)
        return complexity - accuracy
    
    def _find_gradients(self, variables, y=None):
        """
        Calculate gradients using TensorFlow's automatic differentiation.
        
        Args:
            variables: List of variables to calculate gradients for
            y: Optional observation value
        
        Returns:
            List of gradients corresponding to variables
        """
        with tf.GradientTape() as tape:
            vfe = self.get_vfe(y)
        return tape.gradient(vfe, variables)

    def _apply_gradients(self, optimizer, variables, gradients):
        """
        Apply gradients using the specified optimizer.
        
        Args:
            optimizer: The optimizer to use
            variables: List of variables to update
            gradients: List of gradients corresponding to variables
        """
        # Apply gradients using optimizer
        optimizer.apply_gradients(zip(gradients, variables))
        
        # Apply sigma positivity constraints after optimization
        for var in variables:
            if 'sigma' in var.name:
                var.assign(tf.maximum(var, 1e-5))

    def adjust_qx(self):
        gradients = self._find_gradients(self.qx.trainable_variables)
        self._apply_gradients(self.qx_optimizer, self.qx.trainable_variables, gradients)

    def adjust_pyx(self):
        variables = self.pyx.trainable_variables + [self.pypx_sigma]
        gradients = self._find_gradients(variables)
        self._apply_gradients(self.pyx_optimizer, variables, gradients)

    def adjust_px(self):
        gradients = self._find_gradients(self.px.trainable_variables)
        self._apply_gradients(self.px_optimizer, self.px.trainable_variables, gradients)

    def step(self):
        self.adjust_qx()
        self.adjust_pyx()
        self.adjust_px()
        self.world.move()
    

    
World = World()
Agent = Agent(world=World)
complexity = Agent._get_complexity()
accuracy = Agent._get_accuracy(World.observe())
print(f"Complexity: {complexity}")
print(f"Accuracy: {accuracy}")
print(f"VFE: {Agent.get_vfe()}")
print(f"Surprise: {Agent.get_surprise()}")
print("--------------------------------")
for i in range(1000):
    Agent.step()

print(f"Agent qx.mu: {Agent.qx.mu}, Agent qx.sigma: {Agent.qx.sigma}")
print(f"Agent pyx.b0: {Agent.pyx.b0}, Agent pyx.b1: {Agent.pyx.b1}, Agent pypx_sigma: {Agent.pypx_sigma}")
print(f"Agent px.mu: {Agent.px.mu}, Agent px.sigma: {Agent.px.sigma}")
complexity = Agent._get_complexity()
accuracy = Agent._get_accuracy(World.observe())
print(f"Complexity: {complexity}")
print(f"Accuracy: {accuracy}")
print(f"VFE: {Agent.get_vfe()}")
print(f"Surprise: {Agent.get_surprise()}")





