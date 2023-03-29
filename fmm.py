import numpy as np

class Generator:
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
	def sample(self, delta, T, start_values, *args, **kwargs):
		return np.array(start_values)

class BSGenerator(Generator):
	def __init__(self, mu, sigma, cov, r, *args, **kwargs):
		super().__init__(Generator, *args, **kwargs)
		self.mu = mu
		self.sigma = sigma
		self.cov = cov
		self.r = r
	
	def sample(self, delta, T, start_values, *args, **kwargs):
		return np.ones((4, 4))
