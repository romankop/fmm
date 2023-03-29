import numpy as np

class Generator:
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
	def sample(delta, T, start_values, *args, **kwargs):
		return np.array(start_values)

class BSGenerator(Generator):
	def __init__(self, mu, sigma, cov, r):
		super(Generator, *args, **kwargs)
		self.mu = mu
		self.sigma = sigma
		self.cov = cov
		self.r = r
	
	def sample(delta, T, start_values, *args, **kwargs):
		return np.zeros((4, 4))