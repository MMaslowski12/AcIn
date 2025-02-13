# This file has been split into two files:
# - agent.py: Contains the Agent class implementation
# - run_simulation.py: Contains the code to run the simulation

# Please use those files instead of this one.

from normal import Normal, Linear
from World import World
import numpy as np

'''
Experiment to see if this generally works
Get better learning rates and optimizers
Decide what to do with the fact that I model it with easy normals (is it okay because ill switch to nns anyway?)
jump to quadratics

'''

class Agent:
    def __init__(self, world):
        self.px = Normal(0, 1)
        self.qx = Normal(self.px.mu, self.px.sigma) #Without any data, get the most probable x
        self.pyx = Linear(0, 5)
        self.pypx_sigma = 1
        self.world = world

    def get_px(self, x):
        return self.px.pdf(x)

    def get_pyx(self, x):
        return Normal(self.pyx.function(x), self.pypx_sigma).pdf(x)

    def qx(self, x):
        return self.qx.pdf(x)

    def _get_complexity(self):
        return self.qx.kl_divergence(self.px)

    def _get_accuracy(self, y):
        return -0.5 * np.log(2 * np.pi) - np.log(self.pypx_sigma) - \
            0.5 * ((y - self.pyx.function(self.qx.mu))**2 + self.qx.sigma**2) / self.pypx_sigma**2
    
    
    def _get_accuracy(self, y):
        # Calculate expected log likelihood under q(x)
        # For normal distributions, this is -0.5*log(2*pi) - log(sigma) - 0.5*(mu_diff^2 + sigma^2)/sigma^2

        '''
        line 1: constant term + penalizing how indefinite we are about the prediction
        Line 2: How far away p(y given x) is from actual y + sigma of the prediction, divided by how indefinite we are about the prediction
        '''
        mu_function = self.pyx.function(self.qx.mu)
        sigma_function = self.qx.sigma*self.pyx.b1

        return -0.5 * np.log(2*np.pi) - np.log(self.pypx_sigma) - \
               0.5 * ((y - mu_function)**2 + sigma_function**2) / self.pypx_sigma**2 
    
    def get_surprise(self, y=None):
        """
        Computes the expected surprise, i.e., the expected value of -ln p(y) under the generative model.

        Given that both p(y|x) and p(x) are normal distributions, we can calculate this analytically.
        """
        if y is None:
            y = self.world.observe()

        # p(y|x) is a normal distribution with mean given by self.pyx.function(x) and variance self.pypx_sigma^2
        # p(x) is a normal distribution with mean self.px.mu and variance self.px.sigma^2

        # The marginal p(y) is also a normal distribution with:
        # mean = self.pyx.function(self.px.mu)
        # variance = self.pypx_sigma^2 + (self.pyx.b1**2) * self.px.sigma**2

        mean_y = self.pyx.function(self.px.mu)
        variance_y = self.pypx_sigma**2 + (self.pyx.b1**2) * self.px.sigma**2

        # Calculate the surprise as -ln p(y) where p(y) is a normal distribution
        surprise = 0.5 * np.log(2 * np.pi * variance_y) + 0.5 * ((y - mean_y)**2 / variance_y)
        return surprise
        
    
    def get_vfe(self, y=None): 
        if y is None:
            y = self.world.observe()

        complexity = self._get_complexity()
        accuracy = self._get_accuracy(y)
        return complexity - accuracy
    
    def adjust_qx(self):
        learning_rate_mu = 1
        learning_rate_sigma = 1

        # Numerical gradient descent update for mu
        original_mu = self.qx.mu
        self.qx.mu += 1e-5
        vfe_plus = self.get_vfe()
        self.qx.mu = original_mu - 1e-5
        vfe_minus = self.get_vfe()
        grad_mu = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.qx.mu = original_mu - learning_rate_mu * grad_mu

        # Numerical gradient descent update for sigma
        original_sigma = self.qx.sigma
        self.qx.sigma += 1e-5
        vfe_plus = self.get_vfe()
        self.qx.sigma = original_sigma - 1e-5
        vfe_minus = self.get_vfe()
        grad_sigma = (vfe_plus - vfe_minus) / (2 * 1e-5)

        self.qx.sigma = original_sigma - learning_rate_sigma * grad_sigma

        # Ensure sigma remains positive
        if self.qx.sigma <= 0:
            self.qx.sigma = 2e-5

    def adjust_p(self):
        self.adjust_pyx()
        self.adjust_px()

    def adjust_pyx(self):
        learning_rate = 1e-6

        # Numerical gradient descent update for b0
        original_b0 = self.pyx.b0
        self.pyx.b0 += 1e-5
        vfe_plus = self.get_vfe()
        self.pyx.b0 = original_b0 - 1e-5
        vfe_minus = self.get_vfe()
        grad_b0 = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.pyx.b0 = original_b0 - learning_rate * grad_b0

        # Numerical gradient descent update for b1
        original_b1 = self.pyx.b1
        self.pyx.b1 += 1e-5
        vfe_plus = self.get_vfe()
        self.pyx.b1 = original_b1 - 1e-5
        vfe_minus = self.get_vfe()
        grad_b1 = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.pyx.b1 = original_b1 - learning_rate * grad_b1

        # Adjust sigma of pyx
        original_sigma = self.pypx_sigma
        self.pypx_sigma += 1e-5
        vfe_plus = self.get_vfe()
        self.pypx_sigma = original_sigma - 1e-5
        vfe_minus = self.get_vfe()
        grad_sigma = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.pypx_sigma = original_sigma - learning_rate * grad_sigma

        # Ensure sigma remains positive
        if self.pypx_sigma <= 0:
            self.pypx_sigma = 1e-6
        
        
    def adjust_px(self):
        learning_rate = 1e-6

        # Numerical gradient descent update for mu
        original_mu = self.px.mu
        self.px.mu += 1e-5
        vfe_plus = self.get_vfe()
        self.px.mu = original_mu - 1e-5
        vfe_minus = self.get_vfe()
        grad_mu = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.px.mu = original_mu - learning_rate * grad_mu  

        # Numerical gradient descent update for sigma
        original_sigma = self.px.sigma
        self.px.sigma += 1e-5
        vfe_plus = self.get_vfe()
        self.px.sigma = original_sigma - 1e-5
        vfe_minus = self.get_vfe()
        grad_sigma = (vfe_plus - vfe_minus) / (2 * 1e-5)
        self.px.sigma = original_sigma - learning_rate * grad_sigma

        # Ensure sigma remains positive
        if self.px.sigma <= 0:
            self.px.sigma = 1e-6

    def step(self):
        self.adjust_qx()
        self.adjust_p()
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





