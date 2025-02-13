import numpy as np

class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def kl_divergence(self, other):
        """Calculate KL divergence between this normal and another normal distribution"""
        return (
            (self.sigma**2 + (self.mu - other.mu)**2) / (2 * other.sigma**2)
            - 0.5
            + np.log(other.sigma / self.sigma)
        )
            

    def pdf(self, x):
        """Calculate probability density at point x"""
        return (
            1 / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - self.mu) / self.sigma)**2)
        )

class Linear:
    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b1

    def function(self, x):
        return self.b0 + self.b1*x
