from normal import Normal, Linear

class Agent:
    def __init__(self):
        self.px = Normal(0, 1)
        self.qx = Normal(0, 1)
        self.pyx = Linear(0, 1)
        self.pypx_sigma = 1
    
    def get_pyx(self, x):
        return Normal(self.pyx.function(x), self.pypx_sigma).pdf(x)

    def get_px(self, x):
        return self.px.pdf(x)

    def qx(self, x):
        return self.qx.pdf(x)

    def _get_complexity(self):
        return self.qx.kl_divergence(self.px)

    def _get_accuracy(self):
        # Calculate expected log likelihood under q(x)
        # For normal distributions, this is -0.5*log(2*pi) - log(sigma) - 0.5*(mu_diff^2 + sigma^2)/sigma^2
        return -0.5 * np.log(2*np.pi) - np.log(self.pypx_sigma) - \
               0.5 * ((self.pyx.function(self.qx.mu) - self.qx.mu)**2 + self.qx.sigma**2) / self.pypx_sigma**2

    def get_vfe(self): 
        complexity = self._get_complexity()
        accuracy = self._get_accuracy()
        return complexity - accuracy
    

Agent = Agent()
print(Agent.get_vfe())


