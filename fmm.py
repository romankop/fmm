import numpy as np

class Generator:
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
	def sample(self, delta, T, start_values, *args, **kwargs):
		return np.array(start_values)

class BSGenerator(Generator):
  """
  Black-Scholes Generator:
  dS_t = \mu S_t dt + \sigma S_t dW_t
  
  """

  def __init__(self, n=1, mu=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.mu = mu
    if isinstance(self.mu, np.ndarray):
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, list):
      self.mu = np.array(self.mu)
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, (int, float)):
      self.mu = np.full(self.n, self.mu)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.mu * delta * S[:, i - 1, :] + \
                   w[:, i - 1, :] @ self.sigma.T * S[:, i - 1, :] * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class BMGenerator(Generator):
  """
  Brownian Motion Generator:
  dS_t = dW_t
  
  """

  def __init__(self, n=1, cov=None, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n))

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + w[:, i - 1, :] * s_delta
    
    return S

class OUGenerator(Generator):
  """
  Ornstein-Uhlenbeck Generator:
  dS_t = a(b - S_t) dt + \sigma dW_t
  
  """

  def __init__(self, n=1, a=1, b=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.a * delta * (self.b - S[:, i - 1, :]) + \
                   w[:, i - 1, :] @ self.sigma.T * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CIRGenerator(Generator):
  """
  Cox–Ingersoll–Ross Generator:
  dS_t = a (b -S_t) dt + \sigma \sqrt{S_t} dW_t
  
  """
  def __init__(self, n=1, a=1, b=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                   self.a * delta * (self.b - S[:, i - 1, :]) + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.sqrt(S[:, i - 1, :]) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CrtdGenerator(Generator):
  """
  Courtadon Model Generator:
  dS_t = a (b -S_t) dt + \sigma S_t dW_t
  
  """
  def __init__(self, n=1, a=1, b=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                   self.a * delta * (self.b - S[:, i - 1, :]) + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      S[:, i - 1, :] * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class HoLeeGenerator(Generator):
  """
  Ho-Lee Model Generator:
  dS_t = \mu dt + \sigma dW_t
  
  """
  def __init__(self, n=1, mu=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.mu = mu
    if isinstance(self.mu, np.ndarray):
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, list):
      self.mu = np.array(self.mu)
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, (int, float)):
      self.mu = np.full(self.n, self.mu)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.mu * delta + \
                   w[:, i - 1, :] @ self.sigma.T * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CEVGenerator(Generator):
  """
  Constant Elasticity of Variance Model Generator:
  dS_t = \mu S_t dt + \sigma {S_t}^{\alpha} dW_t
  
  """
  def __init__(self, n=1, mu=0, alpha=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.mu = mu
    if isinstance(self.mu, np.ndarray):
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, list):
      self.mu = np.array(self.mu)
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, (int, float)):
      self.mu = np.full(self.n, self.mu)
    else:
      raise AttributeError
    
    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.mu * delta * S[:, i - 1, :] + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.power(S[:, i - 1, :], self.alpha) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CIR1980Generator(Generator):
  """
  Cox–Ingersoll–Ross (1980) Generator:
  dS_t = \sigma {S_t}^{\frac{3}{2}} dW_t
  
  """
  def __init__(self, n=1, alpha=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n
    
    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.power(S[:, i - 1, :], 1.5) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CIR1980AlphaGenerator(Generator):
  """
  Cox–Ingersoll–Ross (1980) + CEV Models Generator
  Can be viewed as Deterministic SABR.
  dS_t = \sigma {S_t}^{\alpha} dW_t
  
  """
  def __init__(self, n=1, alpha=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n
    
    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.power(S[:, i - 1, :], alpha) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class DothanGenerator(Generator):
  """
  Dothan Model (or Black Futures Model) Generator:
  dS_t = \sigma S_t dW_t
  
  """
  def __init__(self, n=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      S[:, i - 1, :] * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CKLSGenerator(Generator):
  """
  Chan–Karolyi–Longstaff–Sanders Model Generator:
  dS_t = a (b - S_t) dt + \sigma {S_t}^{\alpha} dW_t
  
  """
  def __init__(self, n=1, a=0, b=1, alpha=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.a * (self.b - S[:, i - 1, :]) * delta + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.power(S[:, i - 1, :], self.alpha) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class MRGenerator(Generator):
  """
  Marsh and Rosenfeld Model Generator:
  dS_t = (b S_t + a {S_t}^{\alpha - 1}) dt + \sigma {S_t}^{\frac{\alpha}{2}} dW_t
  
  """
  def __init__(self, n=1, a=0, b=1, alpha=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.b * S[:, i - 1, :] * delta + \
                   self.a * np.power(S[:, i - 1, :], self.alpha - 1) * delta + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.power(S[:, i - 1, :], self.alpha / 2) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class DKGenerator(Generator):
  """
  Duffie and Kan Single Factor Model Generator:
  dS_t = a (b - S_t) dt + \sqrt{\sigma + \gamma S_t} dW_t
  
  """
  def __init__(self, n=1, a=0, b=1, gamma=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.gamma = gamma
    if isinstance(self.gamma, np.ndarray):
      if self.gamma.ndim != 1 or len(self.gamma) != self.n:
        raise AttributeError
    elif isinstance(self.gamma, list):
      self.gamma = np.array(self.gamma)
      if self.gamma.ndim != 1 or len(self.gamma) != self.n:
        raise AttributeError
    elif isinstance(self.gamma, (int, float)):
      self.gamma = np.full(self.n, self.gamma)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.a * (self.b - S[:, i - 1, :]) * delta + \
                      w[:, i - 1, :] @ np.sqrt(self.sigma.T + \
                                               self.gamma * S[:, i - 1, :]) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class CnstdGenerator(Generator):
  """
  Constantinides Model Generator:
  dS_t = [a (b - S_t) + \gamma S_t^2] dt + (\sigma + \gamma S_t) dW_t
  
  """
  def __init__(self, n=1, a=0, b=1, gamma=1, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.gamma = gamma
    if isinstance(self.gamma, np.ndarray):
      if self.gamma.ndim != 1 or len(self.gamma) != self.n:
        raise AttributeError
    elif isinstance(self.gamma, list):
      self.gamma = np.array(self.gamma)
      if self.gamma.ndim != 1 or len(self.gamma) != self.n:
        raise AttributeError
    elif isinstance(self.gamma, (int, float)):
      self.gamma = np.full(self.n, self.gamma)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + self.a * (self.b - S[:, i - 1, :]) * delta + \
                    self.gamma * np.power(S[:, i - 1, :], 2) * delta + \
                      w[:, i - 1, :] @ self.sigma.T * s_delta + \
                      w[:, i - 1, :] * self.gamma * S[:, i - 1, :] * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class BKGenerator(Generator):
  """
  Black–Karasinski Model Generator:
  dS_t = a (b - \log{S_t}) dt + \sigma \sqrt{S_t} dW_t
  
  """
  def __init__(self, n=1, a=1, b=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                   self.a * delta * (self.b - np.log(S[:, i - 1, :])) + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      np.sqrt(S[:, i - 1, :]) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class BRSGenerator(Generator):
  """
  Brennan and Schwartz Model Generator:
  dS_t = a S_t (b - \log{S_t}) dt + \sigma S_t dW_t
  
  """
  def __init__(self, n=1, a=1, b=0, sigma=None, cov=None,
               r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError
    
    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(self.n)
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.start

    w = np.random.multivariate_normal(mean=np.zeros(self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] + \
                   self.a * S[:, i - 1, :] * \
                   delta * (self.b - np.log(S[:, i - 1, :])) + \
                      w[:, i - 1, :] @ self.sigma.T * \
                      S[:, i - 1, :] * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class BinTreeGenerator(Generator):
  """
  Binary Tree Model Generator:
  S_t = S_0 \cdot u ^{N_u} d^{N_d}
  
  """
  def __init__(self, n=1, sigma=0, r=0, start_values=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.sigma = sigma
    if isinstance(self.sigma, np.ndarray):
      if self.sigma.ndim != 1 or len(self.sigma) != self.n:
        raise AttributeError
    elif isinstance(self.sigma, list):
      self.sigma = np.array(self.sigma)
      if self.sigma.ndim != 1 or len(self.sigma) != self.n:
        raise AttributeError
    elif isinstance(self.sigma, (int, float)):
      self.sigma = np.full(self.n, self.sigma)
    else:
      raise AttributeError
        
    self.start = start_values
    if isinstance(self.start, np.ndarray):
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, list):
      self.start = np.array(self.start)
      if self.start.ndim != 1 or len(self.start) != self.n:
        raise AttributeError
    elif isinstance(self.start, (int, float)):
      self.start = np.full(self.n, self.start)
    else:
      raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)

    time = np.arange(0, T + delta, delta)

    u = np.exp(self.sigma * np.sqrt(delta))
    d = 1 / u
    p = (np.exp(self.r * delta) - d) / (u - d)

    outcomes = []
    for i in range(self.n):
      outcomes.append(np.random.binomial(1, p[i], size=(N, len(time) - 1)))
    outcomes = np.stack(outcomes, axis=-1) 
    outcomes = outcomes * 2 - 1
    
    S = np.ones((N, len(time), self.n)) * self.start

    for i in range(1, len(time)):
      S[:, i, :] = S[:, i - 1, :] * (u ** outcomes[:, i - 1, :])
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class HestonGenerator(Generator):
  """
  Heston Model Generator:
  dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t
  dV_t = a (b - V_t) dt + \sigma \sqrt{V_t} dZ_t
  
  """
  def __init__(self, n=1, mu=0, a=1, b=1, sigma=None, cov=None, corr=None,
               r=0, start_values_S=1, start_values_V=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.mu = mu
    if isinstance(self.mu, np.ndarray):
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, list):
      self.mu = np.array(self.mu)
      if self.mu.ndim != 1 or len(self.mu) != self.n:
        raise AttributeError
    elif isinstance(self.mu, (int, float)):
      self.mu = np.full(self.n, self.mu)
    else:
      raise AttributeError
    
    self.a = a
    if isinstance(self.a, np.ndarray):
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, list):
      self.a = np.array(self.a)
      if self.a.ndim != 1 or len(self.a) != self.n:
        raise AttributeError
    elif isinstance(self.a, (int, float)):
      self.a = np.full(self.n, self.a)
    else:
      raise AttributeError

    self.b = b
    if isinstance(self.b, np.ndarray):
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, list):
      self.b = np.array(self.b)
      if self.b.ndim != 1 or len(self.b) != self.n:
        raise AttributeError
    elif isinstance(self.b, (int, float)):
      self.b = np.full(self.n, self.b)
    else:
      raise AttributeError
    
    self.startS = start_values_S
    if isinstance(self.startS, np.ndarray):
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, list):
      self.startS = np.array(self.startS)
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, (int, float)):
      self.startS = np.full(self.n, self.startS)
    else:
      raise AttributeError
    
    self.startV = start_values_V
    if isinstance(self.startV, np.ndarray):
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, list):
      self.startV = np.array(self.startV)
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, (int, float)):
      self.startV = np.full(self.n, self.startV)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(2 * self.n)
      if corr is not None:
        if isinstance(corr, (int, float)):
          if abs(corr) > 1:
            raise AttributeError
          corr = np.identity(self.n) * corr
        if isinstance(corr, list):
          corr = np.array(corr)
        if isinstance(corr, np.ndarray):
          if corr.ndim == 1:
            if len(corr) != self.n:
              raise AttributeError
            else:
              corr = np.diag(corr)
          if corr.ndim == 2 and \
            any(~np.equal(corr.shape, self.n)):
            raise AttributeError
          if np.abs(corr).max() > 1:
            raise AttributeError 
        else:
          raise AttributeError
      self.cov[self.n:, :self.n] = corr.T
      self.cov[:self.n, self.n:] = corr
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, 2 * self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(2 * self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.startS
    V = np.ones((N, len(time), self.n)) * self.startV

    w = np.random.multivariate_normal(mean=np.zeros(2 * self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      low_V_1 = np.maximum(V[:, i - 1, :], 0)
      V[:, i, :] = V[:, i - 1, :] + self.a * delta * (self.b - low_V_1) + \
                   w[:, i - 1, self.n:] @ self.sigma.T * \
                   np.sqrt(low_V_1) * s_delta
      S[:, i, :] = S[:, i - 1, :] + self.mu * delta * S[:, i - 1, :] + \
                   w[:, i - 1, :self.n] * S[:, i - 1, :] * \
                   np.sqrt(np.maximum(V[:, i, :], 0)) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class ChenGenerator(Generator):
  """
  Chen Model Generator:
  dS_t = k (\theta_t - S_t) dt + \sqrt{V_t} \sqrt{S_t} dW^1_t
  d\theta_t = a_1 (b_1 - \theta_t) dt + \sigma_1 \sqrt{\theta_t} dW^2_t
  dV_t = a_2 (b_2 - V_t) dt + \sigma_2 \sqrt{V_t} dW^3_t
  
  """
  def __init__(self, n=1, k=1, a_1=1, b_1=1, a_2=1, b_2=1,
               sigma_1=None, sigma_2=None,
               cov=None, corr_12=0, corr_13=0, corr_23=0,
               r=0, start_values_S=1, start_values_T=1,
               start_values_V=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.k = k
    if isinstance(self.k, np.ndarray):
      if self.k.ndim != 1 or len(self.k) != self.n:
        raise AttributeError
    elif isinstance(self.k, list):
      self.k = np.array(self.k)
      if self.k.ndim != 1 or len(self.k) != self.n:
        raise AttributeError
    elif isinstance(self.k, (int, float)):
      self.k = np.full(self.n, self.k)
    else:
      raise AttributeError
    
    self.a_1 = a_1
    if isinstance(self.a_1, np.ndarray):
      if self.a_1.ndim != 1 or len(self.a_1) != self.n:
        raise AttributeError
    elif isinstance(self.a_1, list):
      self.a_1 = np.array(self.a_1)
      if self.a_1.ndim != 1 or len(self.a_1) != self.n:
        raise AttributeError
    elif isinstance(self.a_1, (int, float)):
      self.a_1 = np.full(self.n, self.a_1)
    else:
      raise AttributeError

    self.b_1 = b_1
    if isinstance(self.b_1, np.ndarray):
      if self.b_1.ndim != 1 or len(self.b_1) != self.n:
        raise AttributeError
    elif isinstance(self.b_1, list):
      self.b_1 = np.array(self.b_1)
      if self.b_1.ndim != 1 or len(self.b_1) != self.n:
        raise AttributeError
    elif isinstance(self.b_1, (int, float)):
      self.b_1 = np.full(self.n, self.b_1)
    else:
      raise AttributeError

    self.a_2 = a_2
    if isinstance(self.a_2, np.ndarray):
      if self.a_2.ndim != 1 or len(self.a_2) != self.n:
        raise AttributeError
    elif isinstance(self.a_2, list):
      self.a_2 = np.array(self.a_2)
      if self.a_2.ndim != 1 or len(self.a_2) != self.n:
        raise AttributeError
    elif isinstance(self.a_2, (int, float)):
      self.a_2 = np.full(self.n, self.a_2)
    else:
      raise AttributeError

    self.b_2 = b_2
    if isinstance(self.b_2, np.ndarray):
      if self.b_2.ndim != 1 or len(self.b_2) != self.n:
        raise AttributeError
    elif isinstance(self.b_2, list):
      self.b_2 = np.array(self.b_2)
      if self.b_2.ndim != 1 or len(self.b_2) != self.n:
        raise AttributeError
    elif isinstance(self.b_2, (int, float)):
      self.b_2 = np.full(self.n, self.b_2)
    else:
      raise AttributeError
    
    self.startS = start_values_S
    if isinstance(self.startS, np.ndarray):
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, list):
      self.startS = np.array(self.startS)
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, (int, float)):
      self.startS = np.full(self.n, self.startS)
    else:
      raise AttributeError

    self.startT = start_values_T
    if isinstance(self.startT, np.ndarray):
      if self.startT.ndim != 1 or len(self.startT) != self.n:
        raise AttributeError
    elif isinstance(self.startT, list):
      self.startT = np.array(self.startT)
      if self.startT.ndim != 1 or len(self.startT) != self.n:
        raise AttributeError
    elif isinstance(self.startT, (int, float)):
      self.startT = np.full(self.n, self.startT)
    else:
      raise AttributeError
    
    self.startV = start_values_V
    if isinstance(self.startV, np.ndarray):
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, list):
      self.startV = np.array(self.startV)
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, (int, float)):
      self.startV = np.full(self.n, self.startV)
    else:
      raise AttributeError

    if sigma_1 is None:
      self.sigma_1 = np.identity(self.n)
    else:
      self.sigma_1 = sigma_1
      if isinstance(self.sigma_1, (int, float)):
        if self.sigma_1 <= 0:
          raise AttributeError
        self.sigma_1 = self.sigma_1 * np.identity(self.n)
      if isinstance(self.sigma_1, list):
        self.sigma_1 = np.array(self.sigma_1)
      if isinstance(self.sigma_1, np.ndarray):
        if self.sigma_1.ndim == 1:
          if len(self.sigma_1) != self.n:
            raise AttributeError
          else:
            self.sigma_1 = np.diag(self.sigma_1)
        if self.sigma_1.ndim == 2 and \
          any(~np.equal(self.sigma_1.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError
    
    if sigma_2 is None:
      self.sigma_2 = np.identity(self.n)
    else:
      self.sigma_2 = sigma_2
      if isinstance(self.sigma_2, (int, float)):
        if self.sigma_2 <= 0:
          raise AttributeError
        self.sigma_2 = self.sigma_2 * np.identity(self.n)
      if isinstance(self.sigma_2, list):
        self.sigma_2 = np.array(self.sigma_2)
      if isinstance(self.sigma_2, np.ndarray):
        if self.sigma_2.ndim == 1:
          if len(self.sigma_2) != self.n:
            raise AttributeError
          else:
            self.sigma_2 = np.diag(self.sigma_2)
        if self.sigma_2.ndim == 2 and \
          any(~np.equal(self.sigma_2.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(3 * self.n)

      if corr_12 is not None:
        if isinstance(corr_12, (int, float)):
          if abs(corr_12) > 1:
            raise AttributeError
          corr_12 = np.identity(self.n) * corr_12
        if isinstance(corr_12, list):
          corr_12 = np.array(corr_12)
        if isinstance(corr_12, np.ndarray):
          if corr_12.ndim == 1:
            if len(corr_12) != self.n:
              raise AttributeError
            else:
              corr_12 = np.diag(corr_12)
          if corr_12.ndim == 2 and \
            any(~np.equal(corr_12.shape, self.n)):
            raise AttributeError
          if np.abs(corr_12).max() > 1:
            raise AttributeError 
        else:
          raise AttributeError
      
      self.cov[self.n:2 * self.n, :self.n] = corr_12.T
      self.cov[:self.n, self.n:2 * self.n] = corr_12

      if corr_13 is not None:
        if isinstance(corr_13, (int, float)):
          if abs(corr_13) > 1:
            raise AttributeError
          corr_13 = np.identity(self.n) * corr_13
        if isinstance(corr_13, list):
          corr_13 = np.array(corr_13)
        if isinstance(corr_13, np.ndarray):
          if corr_13.ndim == 1:
            if len(corr_13) != self.n:
              raise AttributeError
            else:
              corr_13 = np.diag(corr_13)
          if corr_13.ndim == 2 and \
            any(~np.equal(corr_13.shape, self.n)):
            raise AttributeError
          if np.abs(corr_13).max() > 1:
            raise AttributeError 
        else:
          raise AttributeError
      
      self.cov[2 * self.n:3 * self.n, :self.n] = corr_13.T
      self.cov[:self.n, 2 * self.n:3 * self.n] = corr_13

      if corr_23 is not None:
        if isinstance(corr_23, (int, float)):
          if abs(corr_23) > 1:
            raise AttributeError
          corr_23 = np.identity(self.n) * corr_23
        if isinstance(corr_23, list):
          corr_23 = np.array(corr_23)
        if isinstance(corr_23, np.ndarray):
          if corr_23.ndim == 1:
            if len(corr_23) != self.n:
              raise AttributeError
            else:
              corr_23 = np.diag(corr_23)
          if corr_23.ndim == 2 and \
            any(~np.equal(corr_23.shape, self.n)):
            raise AttributeError
          if np.abs(corr_23).max() > 1:
            raise AttributeError 
        else:
          raise AttributeError
      
      self.cov[2 * self.n:3 * self.n, self.n:2 * self.n] = corr_23.T
      self.cov[self.n:2 * self.n, 2 * self.n:3 * self.n] = corr_23

    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, 3 * self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(3 * self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.startS
    TH = np.ones((N, len(time), self.n)) * self.startT
    V = np.ones((N, len(time), self.n)) * self.startV

    w = np.random.multivariate_normal(mean=np.zeros(3 * self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      low_V_1 = np.maximum(V[:, i - 1, :], 0)
      V[:, i, :] = V[:, i - 1, :] + self.a_2 * delta * (self.b_2 - low_V_1) + \
                   w[:, i - 1, 2 * self.n:] @ self.sigma_2.T * \
                   np.sqrt(low_V_1) * s_delta
      
      low_TH_1 = np.maximum(TH[:, i - 1, :], 0)
      TH[:, i, :] = TH[:, i - 1, :] + self.a_1 * delta * (self.b_1 - low_TH_1) + \
                   w[:, i - 1, self.n:2 * self.n] @ self.sigma_1.T * \
                   np.sqrt(low_TH_1) * s_delta

      S[:, i, :] = S[:, i - 1, :] + \
                   self.k * delta * \
                   (np.maximum(TH[:, i, :], 0) - S[:, i - 1, :]) + \
                   w[:, i - 1, :self.n] * np.sqrt(S[:, i - 1, :]) * \
                   np.sqrt(np.maximum(V[:, i, :], 0)) * s_delta
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S

class SABRGenerator(Generator):
  """
  SABR Model Generator:
  dS_t = V_t {S_t}^{\alpha} dW_t
  dV_t = \sigma V_t dZ_t
  
  """
  def __init__(self, n=1, alpha=1, sigma=None, cov=None, corr=None,
               r=0, start_values_S=1, start_values_V=1, *args, **kwargs):
    super().__init__(Generator, *args, **kwargs)
    
    self.n = n

    self.alpha = alpha
    if isinstance(self.alpha, np.ndarray):
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, list):
      self.alpha = np.array(self.alpha)
      if self.alpha.ndim != 1 or len(self.alpha) != self.n:
        raise AttributeError
    elif isinstance(self.alpha, (int, float)):
      self.alpha = np.full(self.n, self.alpha)
    else:
      raise AttributeError
    
    self.startS = start_values_S
    if isinstance(self.startS, np.ndarray):
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, list):
      self.startS = np.array(self.startS)
      if self.startS.ndim != 1 or len(self.startS) != self.n:
        raise AttributeError
    elif isinstance(self.startS, (int, float)):
      self.startS = np.full(self.n, self.startS)
    else:
      raise AttributeError
    
    self.startV = start_values_V
    if isinstance(self.startV, np.ndarray):
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, list):
      self.startV = np.array(self.startV)
      if self.startV.ndim != 1 or len(self.startV) != self.n:
        raise AttributeError
    elif isinstance(self.startV, (int, float)):
      self.startV = np.full(self.n, self.startV)
    else:
      raise AttributeError

    if sigma is None:
      self.sigma = np.identity(self.n)
    else:
      self.sigma = sigma
      if isinstance(self.sigma, (int, float)):
        if self.sigma <= 0:
          raise AttributeError
        self.sigma = self.sigma * np.identity(self.n)
      if isinstance(self.sigma, list):
        self.sigma = np.array(self.sigma)
      if isinstance(self.sigma, np.ndarray):
        if self.sigma.ndim == 1:
          if len(self.sigma) != self.n:
            raise AttributeError
          else:
            self.sigma = np.diag(self.sigma)
        if self.sigma.ndim == 2 and \
          any(~np.equal(self.sigma.shape, self.n)):
          raise AttributeError
      else:
        raise AttributeError

    self.cov = cov
    if cov is None:
      self.cov = np.identity(2 * self.n)
      if corr is not None:
        if isinstance(corr, (int, float)):
          if abs(corr) > 1:
            raise AttributeError
          corr = np.identity(self.n) * corr
        if isinstance(corr, list):
          corr = np.array(corr)
        if isinstance(corr, np.ndarray):
          if corr.ndim == 1:
            if len(corr) != self.n:
              raise AttributeError
            else:
              corr = np.diag(corr)
          if corr.ndim == 2 and \
            any(~np.equal(corr.shape, self.n)):
            raise AttributeError
          if np.abs(corr).max() > 1:
            raise AttributeError 
        else:
          raise AttributeError
      self.cov[self.n:, :self.n] = corr.T
      self.cov[:self.n, self.n:] = corr
    else:
      if isinstance(self.cov, list):
        self.cov = np.array(self.cov)
      if not isinstance(self.cov, np.ndarray):
        raise AttributeError
      if self.cov.ndim != 2 or any(~np.equal(self.cov.shape, 2 * self.n)):
        raise AttributeError
      if any(np.diag(self.cov) != np.ones(2 * self.n)) or \
         np.abs(self.cov).max() > 1:
        raise AttributeError
    
    self.r = r

  def B(self, time_end, time_start=0, **kwargs):

    return np.exp(self.r * (time_end - time_start))
	
  def sample(self, delta, T, N, include_rf=False, *args, **kwargs):
    
    T = int(T / delta) * delta
    s_delta = np.sqrt(delta)
    
    time = np.arange(0, T + delta, delta)
    
    S = np.ones((N, len(time), self.n)) * self.startS
    V = np.ones((N, len(time), self.n)) * self.startV

    w = np.random.multivariate_normal(mean=np.zeros(2 * self.n),
                                      cov=self.cov, size=(N, len(time) - 1))

    for i in range(1, len(time)):
      low_V_1 = np.abs(V[:, i - 1, :])
      V[:, i, :] = V[:, i - 1, :] + \
                   w[:, i - 1, self.n:] @ self.sigma.T * low_V_1 * s_delta
      S[:, i, :] = S[:, i - 1, :] + \
                   w[:, i - 1, :self.n] * np.abs(V[:, i, :]) * s_delta * \
                    np.power(S[:, i - 1, :], self.alpha)
    
    if include_rf:
      B = self.B(time) * np.ones((N, len(time), 1))
      return np.concatenate([B, S], axis=-1)

    return S
