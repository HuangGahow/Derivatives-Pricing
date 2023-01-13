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
    """
    Binary option pricing:
    
    parameters
    ====================
    S0: stock price
    K: strike price
    r: risk free rate
    q: dividend rate
    T: time to expiry
    sig: volatility
    cp_flag: 'put' or 'call'
    ====================
    
    
    returns
    ====================
    Close form price
    Monte carlo method price, and standard error 
    PDE method price
    ====================
    """
    def __init__(self, S0, K, r, T, sigma, cp_flag = 'call'):
        
        self.S0 = S0            #stock price
        self.K = K              #strike price
        self.r = r              #risk-free rate
        self.T = T              # expiry
        self.sigma = sigma      # volatility of stock
        self.cp_flag = cp_flag  # call or put
        
    def get_price_CF(self):
        """CF: close solution form"""
        self.d2 = (log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2)*self.T
                   ) / (self.sigma * sqrt(self.T))
        
        if self.cp_flag == 'call':
            CF_price = exp(-self.r * self.T)* N(self.d2)
        elif self.cp_flag == 'put':
            CF_price = exp(-self.r * self.T)* N(1 - self.d2)
            
            
        return CF_price
    
    def get_price_MC(self, N_simulation=20000):
        """MC: Monte Carlo"""
        
        W = (r - sigma**2/2)*T  + ss.norm.rvs(loc=0, scale=sigma, size=N_simulation)
        S_T = S0 * np.exp(W)
        MC_price = np.exp(-r*T) * np.mean( S_T > K )
        MC_std = np.exp(-r*T) * ss.sem( S_T > K )
        
        
        return MC_price, MC_std
    
    
        

# In[1]

"""Option variables"""
S0 = 100.0              # spot stock price
K = 100.0               # strike
T = 1.0                 # maturity 
r = 0.1                 # risk free rate 
sigma = 0.2               # diffusion coefficient or volatility

gahow = Binary_option(S0, K, r, T, sigma)
gahow.get_price_MC(N_simulation=200000)


# In[2]
"""plot relation between stock price and option price"""
Nspace = 6000   # M space steps
Ntime = 6000    # N time steps   
S_max = 3*float(K)                
S_min = float(K)/3
x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)    # space discretization
T_array, dt = np.linspace(0, T, Ntime, retstep=True)       # time discretization


# In[3]
Payoff = np.where(np.exp(x)>K, 1, 0) 
S = np.exp(x)

def currying_add(func):
    def wrapper(S0, K=100.0, r=0.1, T=1.0, sigma=0.2):
        return func(S0, K, r, T, sigma)
    return wrapper

@currying_add
def getprice(S0, K, r, T, sigma):
    return Binary_option(S0, K, r, T, sigma).get_price_CF()
 
V = list(map(getprice, S))



fig = plt.figure()
ax1 = fig.add_subplot()
ax1.plot(S, Payoff, color='blue',label="Payoff")
ax1.plot(S, V, color='red',label="Binary curve")
ax1.set_xlim(60,200); ax1.set_ylim(0,1.1)
ax1.set_xlabel("S"); ax1.set_ylabel("V"); ax1.legend(loc='upper left')
ax1.set_title("Curve at t=0")
