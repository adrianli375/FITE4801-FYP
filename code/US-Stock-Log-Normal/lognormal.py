#region imports
from AlgorithmImports import *
#endregion
import numpy as np
from scipy.stats import norm as normal


class LogNormal:
    def __init__(self, y: np.array, fit=True, **kwargs):
        self.y = y
        self.n = y.shape[0]
        self.mu = None
        self.sigma = None
        if fit:
            self.fit(**kwargs)
    
    def fit(self, method="MLE"):
        y = self.y
        n = self.n
        # fit the parameters using either maximum likelihood estimation (MLE) 
        # or method of moments (MM)
        if method == 'MLE':
            self.mu = 1 / n * np.log(y).sum()
            self.sigma = 1 / (n - 1) * ((np.log(y) - self.mu) ** 2).sum()
        elif method == 'MM':
            y_bar = 1 / n * y.sum()
            s_squared = 1 / (n - 1) * ((y - y_bar) ** 2).sum()
            self.sigma = np.sqrt(np.log(s_squared / (y_bar) ** 2 + 1))
            self.mu = np.log(y_bar / np.exp(1 / 2 * self.sigma ** 2))
    
    def calculate_value_at_risk(self, alpha: float):
        assert 0 < alpha < 1
        return np.exp(normal.ppf(alpha) * self.sigma + self.mu)
    
    def calculate_tail_value_at_risk(self, alpha: float):
        assert 0 < alpha < 1
        return np.exp(self.mu + 1 / 2 * self.sigma ** 2) * normal.cdf(self.sigma - normal.ppf(alpha)) / (1 - alpha)