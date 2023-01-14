# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:23:37 2022

@author: HP
"""
from scipy.stats import norm
import numpy as np
import pandas as pd
n = norm.pdf # pdf
N = norm.cdf # cdf

'''
IV = implied volatility
S = stock price
K = strike price
T = time to maturity
r = discount rate(risk free rate)
v = volatility
q = dividend rate
V = price of option
'''
# In[1]
class IV():
    def __init__(self, S, K, market_price, T, r, flag, initial_value, iterations, precision):
        self.S = S
        self.K = K
        self.market_price = market_price
        self.T = T
        self.r = r
        self.flag = flag
        self.initial_value = initial_value
        self.iterations = iterations
        self.precision = precision

    def Get_BS_price(self,v,q=0.0):
        d1 = (np.log(self.S/self.K )+(self.r+0.5*v**2)*self.T)/(v*np.sqrt(self.T))
        d2 = d1 - v*np.sqrt(self.T)
        if flag == 'call':
            V = self.S*np.exp(-q*self.T)*N(d1)-self.K*np.exp(-self.r*self.T)*N(d2)
        else:
            V = self.K*np.exp(-r*self.T)*N(-d2)-self.S*np.exp(-q*self.T)*N(-d1)
    
        return V

    def Get_BS_vega(self,v):
        d1= (np.log(self.S/self.K)+(self.r+v** 2/2.)*self.T)/(v*np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T)*n(d1)
        return vega

    def Get_BS_IV(self):
        """initial_value is initial volatility we set"""
        initial_value1 = self.initial_value
        for i in range(self.iterations):
            model_price = self.Get_BS_price(initial_value1)
            vega = self.Get_BS_vega(initial_value1)
            diff = self.market_price - model_price  # our root

            if (abs(diff) < self.precision):
                return initial_value1
            initial_value1 = initial_value1 + diff/vega # f(x) / f'(x)

        # value wasn't found, return best guess so far
        return initial_value1
    
# In[2]

"""
S:AAPL's price in 2023/01/13
r:1 year treasury rate (https://ycharts.com/indicators/1_year_treasury_rate)
K:Strike price of AAPL's options (https://www.barchart.com/stocks/quotes/AAPL/options?expiration=2023-01-13)
"""
market_price=[9.55, 8.55, 7.53, 6.55, 5.55, 4.53, 3.58,2.38, 1.86,1.18, 0.66, 0.33, 0.15, 0.07, 0.03,0.02,0.01,0.01,0.01,0.01]
K=np.arange(124, 143, 1)
pairs = zip(market_price, K)
T=46/365
S=133.41
r= 4.66/100
flag = 'call'
initial_value = 0.5
iterations = 1000
precision = 1.0e-7
import matplotlib.pyplot as plt
# get implied volatility of all strike price
result = []
for m, k in paris:
    iv1 = IV(S, k, m, T, r, flag, initial_value, iterations, precision)
    sigma1 =iv1.Get_BS_IV()
    result.append(sigma1)

plt.plot(K,result)
plt.show()
