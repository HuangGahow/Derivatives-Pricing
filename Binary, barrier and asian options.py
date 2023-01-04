# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:23:35 2023

@author: HP
"""

"""
Binary(digital) option
"""
import numpy as np
import scipy as scp
import scipy.stats as ss
import matplotlib.pyplot as plt
from math import exp, log, sqrt
N = ss.norm.cdf
n = ss.norm.pdf

# In[0]
class Binary_option():
    
    def __init__(self, S0, K, r, T, sigma, cp_flag = 'call'):
        
        self.S0 = S0 #stock price
        self.K = K  #strike price
        self.r = r  #risk-free rate
        self.T = T  # expiry
        self.sigma = sigma # volatility of stock
        self.cp_flag = cp_flag # call or put
        
    def get_price_CF(self):
        """CF means close form"""
        self.d2 = (log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2)*self.T
                   ) / (self.sigma * sqrt(self.T))
        if self.cp_flag == 'call':
            price_CF = exp(-self.r * self.T)* N(self.d2)
        elif self.cp_flag == 'put':
            price_CF = exp(-self.r * self.T)* N(1 - self.d2)
        return price_CF
    
    def get_price_MC(self, N_simulation=20000):
        """MC means Monte Carlo"""
        W = (r - sigma**2/2)*T  + ss.norm.rvs(loc=0, scale=sigma, size=N_simulation)
        S_T = S0 * np.exp(W)
        price_MC = np.exp(-r*T) * np.mean( S_T > K )
        digital_std_err = np.exp(-r*T) * ss.sem( S_T > K )
        return price_MC


# In[1]

"""Option variables"""
S0 = 100.0              # spot stock price
K = 100.0               # strike
T = 1.0                 # maturity 
r = 0.1                 # risk free rate 
sigma = 0.2               # diffusion coefficient or volatility

gahow = Binary_option(S0, K, r, T, sigma, 'put')
gahow.get_price_MC(N_simulation=200000)

# In[2]
"""continuing"""