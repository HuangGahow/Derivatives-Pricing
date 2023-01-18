# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:44:33 2023

@author: HP
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu

N = ss.norm.cdf
n = ss.norm.pdf
# In[1]
class Barrier_option():
    """
    Barrier option pricing:
    
    parameters
    ====================
    S0: stock price
    K: strike price
    r: risk free rate
    q: dividend rate
    T: time to expiration
    sig: volatility
    cp_flag: 'put' or 'call'
    bartype:'up-out' or 'down-in'
    ====================
    
    
    returns
    ====================
    get_price_CF: Close form price
    get_price_MC: Monte carlo method price, and standard error 
    get_price_PDE: PDE method price
    ====================
    """
    def __init__(self, S0, K, r, T, sigma, B, cp_flag ='down-out-call', bartype = 'down-out'):
        
        self.S0 = S0
        self.K = K              
        self.r = r             
        self.T = T              
        self.sigma = sigma     
        self.B = B              
        self.cp_flag = cp_flag        
        self.bartype = bartype
        
    def get_price_CF(self):
        d1 = lambda t,s: 1/(self.sigma*np.sqrt(t)) * ( np.log(s) + (self.r+self.sigma**2/2)*t ) 
        d2 = lambda t,s: 1/(self.sigma*np.sqrt(t)) * ( np.log(s) + (self.r-self.sigma**2/2)*t )
        
        if self.bartype == 'up-out' & self.cp_flag == 'call':
            CF_price = self.S0 * ( N( d1(self.T, self.S0/self.K) ) - N(d1(self.T, self.S0/self.B))) \
                        - np.exp(-self.r*self.T) * self.K * (N(d2(self.T, self.S0/self.K) )- N(d2(self.T, self.S0/self.B))) \
                            - self.B * (self.S0/self.B)**(-2*self.r/self.sigma**2) * (N(d1(self.T, self.B**2/(self.S0*K))) - N(d1(self.T, self.B/self.S0))) \
                                + np.exp(-self.r*self.T) * self.K * (self.S0/self.B)**(-2*self.r/self.sigma**2 + 1) * \
                                    (N(d2(self.T, self.B**2/(self.S0*self.K) ))- N(d2(self.T, self.B/self.S0)))
        elif self.bartype = 'down-in' & self.cp_flag == 'call':
            CF_price = self.S0 * (self.B/S0)**(1+2*self.r/self.sigma**2) * (N(d1(self.T, self.B**2/(self.S0*self.K)))) \
                    - np.exp(-self.r*self.T) * self.K * (self.B/self.S0)**(-1+2*self.r/self.sigma**2) * (N(d2(self.T, self.B**2/(self.S0*self.K))))
        
        return CF_price
    
    def get_price_MC(self, N_simulation = 20000, N_step = 10000):
        self.N_simulation = N_simulation
        self.N_step = 10000
        dt = self.T/(self.N_step - 1)
        X_0 = np.zero((self.N_simulation, 1))
        increments = np.random.normal(loc = dt*(self.r - 1/2 * self.sigma**2), scale = np.sqrt(dt) * self.sigma, size = (N_simulation, N_step - 1))
        
        X = np.concatenate(X_0, increments, axis =1).cumsum(axis = 1)
        S = S0 * np.exp(X)
        
        M = np.max(S, axis = 1)
        MM = np.min(S, axis = 1) 
        
        if self.bartype == 'up-out' & self.cp_flag == 'call':
            MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum(S[:, -1] - self.K, 0) @ (M < self.B))
        elif self.bartype = 'down-in' & self.cp_flag == 'call':
            MC_price = np.exp(-self.r * self.T) * np.mean(np.maximum(S[:, -1] - self.K, 0 ) @ (MM <= self.B))
            
        return MC_price
    
    def get_price_PDE(self, Nstep = 5000, Ntime =4000 ):
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        if self.bartype == 'up-out' & self.cp_flag == 'call':
            S_max = self.B
            S_min = 1/3 * np.minimum(self.S0, self.B)
             
        
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        
        x_vec, dx = np.linspace(x_min, x_max, num = self.Nstep, retstep = True)
        t_vec, dt = np.linsapce(0, self.T, num = self.Ntime, restep = True)
        
        self.V[:, -1] = np.maximum(np.exp(x_vec) - self.K, 0)
        self.V[-1, :] = 0
        self.V[0, :] = 0
        
        offset = np.zeros((self.Nstep - 2))
        sigma2 = self.sigma ** 2
        dxx = dx **2
    
        #coefficient of matrix
        a = ( (dt/2) * ( (self.r-0.5*sigma2)/dx - sigma2/dxx ) )
        b = ( 1 + dt * ( sigma2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape = (self.Nstep-2, self.Nstep-2), format = 'csc')
        
        DD = splu(self.D)
        
        for i in range(Nstep-2, -1, -1):
            offset[0] = a * self.V[0, i]
            offset[-1] = c * self.V[-1,i]
            self.V[1:-1, i] = DD.solve(self.V[1:-1, i+1] - offset)
        
        return np.interp(x0, x_vec, self.V[:,0])
        