# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:23:35 2023

@author: HP
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu

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
        self.omega = 1 if cp_flag =='call' else -1
        
    def get_price_CF(self):
        """CF: close solution form"""
        self.d2 = (np.log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2)*self.T
                   ) / (self.sigma * np.sqrt(self.T))
        
        if self.cp_flag == 'call':
            CF_price = np.exp(-self.r * self.T)* N(self.d2)
        elif self.cp_flag == 'put':
            CF_price = np.exp(-self.r * self.T)* N(1 - self.d2)
            
            
        return CF_price
    
    def get_price_MC(self, N_simulation=20000):
        """MC: Monte Carlo"""
        
        W = (self.r - self.sigma**2/2)*T  + ss.norm.rvs(loc=0, scale=self.sigma, size=N_simulation)
        S_T = self.S0 * np.exp(W)
        if self.cp_flag == 'call':
            MC_price = np.exp(-self.r*T) * np.mean(S_T > self.K )
            MC_std = np.exp(-r*T) * ss.sem( S_T > self.K )
        elif self.cp_flag == 'put':
            MC_price = np.exp(-self.r*T) * np.mean(S_T < self.K )
            MC_std = np.exp(-r*T) * ss.sem( S_T < self.K )
        
        
        return MC_price, MC_std
    
    def get_price_PDE(self, Nstep = 5000, Ntime = 4000):
        """PDE Method"""
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.maximum(self.S0, self.K)
        
        x0 = np.log(self.S0)
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        
        x_vec, dx = np.linspace(x_min, x_max, num = self.Nstep, retstep = True)
        t_vec, dt = np.linspace(0, self.T, num = self.Ntime, retstep = True)
        
        self.V = np.zeros((self.Nstep, self.Ntime))
        
        if self.cp_flag == 'call':
            self.V[:, -1] = np.where(np.exp(x_vec) > self.K, 1, 0)
            self.V[0, :] = 0
            self.V[-1, :] = 1
        elif self.cp_flag == 'put':
            self.V[:, -1] = np.where(np.exp(x_vec) < self.K, 1, 0)
            self.V[0, :] = 1
            self.V[-1, :] = 0
            
            
        offset = np.zeros((self.Nstep - 2))
        sigma2 = self.sigma ** 2
        dxx = dx **2
    
        #coefficient of matrix
        a = ( (dt/2) * ( (self.r-0.5*sigma2)/dx - sigma2/dxx ) )
        b = ( 1 + dt * ( sigma2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(self.Nstep-2, self.Nstep-2)).tocsc()
        DD = splu(self.D)
        
        for i in range(self.Ntime-2, -1, -1):
            offset[0] = a * self.V[0, i]
            offset[-1] = c * self.V[-1,i]
            self.V[1:-1, i] = DD.solve(self.V[1:-1, i+1] - offset)
        
        return np.interp(x0, x_vec, self.V[:,0])
        
    
    
        

# In[1]

"""Option variables"""
S0 = 100.0              # spot stock price
K = 100.0               # strike
T = 1.0                 # maturity 
r = 0.1                 # risk free rate 
sigma = 0.2               # diffusion coefficient or volatility

A = Binary_option(S0, K, r, T, sigma)
print('CF price:',A.get_price_CF())
print('MC price:',A.get_price_MC())
print('PDE price:',A.get_price_PDE())

# In[2]
"""plot relation between stock price and option price----two dimension"""
Nspace = 5000   # M space steps
Ntime = 4000    # N time steps   
S_max = 3*float(K)                
S_min = float(K)/3
x_max = np.log(S_max)  # A2
x_min = np.log(S_min)  # A1

x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)    # space discretization
T_vec, dt = np.linspace(0, T, Ntime, retstep=True)       # time discretization

Payoff = np.where(np.exp(x)>K, 1, 0) 
S = np.exp(x)
#just try to use currying
def currying_add(func):
    def wrapper(S0, K=100.0, r=0.1, T=1.0, sigma=0.2):
        return func(S0, K, r, T, sigma)
    return wrapper

@currying_add
def getprice(S0, K, r, T, sigma):
    return Binary_option(S0, K, r, T, sigma).get_price_CF()
 
V = list(map(getprice, S))
ax1 =plt.figure()
ax1.plot(S, Payoff, color='blue',label="Payoff")
ax1.plot(S, V, color='red',label="Binary curve")
ax1.set_xlim(60,200); ax1.set_ylim(0,1.1)
ax1.set_xlabel("S"); ax1.set_ylabel("V"); ax1.legend(loc='upper left')
ax1.set_title("Curve at t=0")

# In[3]
"""plot relation between stock price, time to expiration and option price----3 dimension"""
X, Y = np.meshgrid(T_vec, S) #Reference: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
fig = plt.figure()
ax2 = fig.subplots(subplot_kw={"projection": "3d"}) #Reference: https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html#sphx-glr-plot-types-3d-surface3d-simple-py
ax2.plot_surface(Y, X, A.V)
ax2.set_title("Digital option surface"); ax2.set_xlabel("S"); ax2.set_ylabel("t"); ax2.set_zlabel("V")
plt.show()
