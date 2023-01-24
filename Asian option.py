# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:02:17 2023

@author: HP
"""
import numpy as np
import scipy as scp
import scipy.stats as ss
import matplotlib.pyplot as plt
from math import exp, log, sqrt
N = ss.norm.cdf
n = ss.norm.pdf

# In[0]
class Asian_option():
    """
    Asian option pricing: only European style, and Arithmetic Average style.
    
    
    
    parameters
    ====================
    S0: stock price
    K: strike price
    r: risk free rate
    q: dividend rate
    T: time to expiry
    sig: volatility
    cp_flag: 'put' or 'call'
    strike_flag: 'fix'(fix strike) or 'float'(floating strike)
    
    ###To be developed###
    avr_flag: 'ari'(arithmetic average) or 'geo'(geomatric average)
    ====================
    
    
    returns
    ====================
    Monte carlo method price and standard error 
    PDE method price
    ====================
    
    """
    def  __init__(self, S0, K, r, q, T, sigma, cp_flag, strike_flag):
        
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.T = T
        self.sigma = sigma
        self.cp_flag = cp_flag
        self.strike_flag = strike_flag
        
    def get_price_MC(self, self.Nstep = 3000, path = 5000):
        # Simulate stock price's paths by MC
        self.Nstep = Nstep 
        self.path = path
        dt = self.T / (self.Nstep - 1)
        X_0 = np.zeros((self.path, 1))
        increments = np.random.normal(loc=(self.r - self.sigma ** 2 /2)*dt, scale = np.sqrt(dt) * self.sigma,
                                      size = (paths, Nsteps-1))
        X = np.concatenate((X_0, increments), axis=1).cumsum(1)
        S = S0 * exp(X)
        
        
        
        
        if strike_flag == 'fix':
            if cp_flag == 'call':
                MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum( A - self.K, 0))
                MC_std = np.exp(-self.r * self.T) * np.std(np.maximum( A - self.K, 0))
            elif cp_flag == 'put':
                MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum( self.K - A, 0))
                MC_std = np.exp(-self.r * self.T) * np.std(np.maximum( self.K - A, 0))
                
                
        elif strike_flag == 'float':
            if cp_flag == 'call':
                MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum( S[:,-1] - A, 0))
                MC_std = np.exp(-self.r * self.T) * np.std(np.maximum( S[:,-1] - A, 0))
            elif cp_flag == 'put':
                MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum( A - S[:,-1], 0))
                MC_price = np.exp(-self.r * self.T) * np.std(np.maximum( A - S[:,-1], 0))
        
        print("MC_price: ", MC_price)
        print('with std: ', MC_std)
        
        return MC_price, MC_std
    
    def get_price_PDE(self):
        
        if strike_flag == 'fix':
            
            
            
    
                
        
        
        
        
    
import time
# In[2]
%%time
np.random.seed(seed=41)
r=0.5
T= 1
sig =0.2
Nsteps = 10000
paths = 50000
dt = T/(Nsteps-1) 
X_0 = np.zeros((paths,1))
# In[4]
%%time
increments = ss.norm.rvs(loc=(r - sig**2/2)*dt, scale=np.sqrt(dt)*sig, size=(paths,Nsteps-1))
# In[3]
%%time
y = np.random.normal(loc=(r - sig**2/2)*dt, scale=np.sqrt(dt)*sig, size=(paths,Nsteps-1))
