# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:12:58 2022

@author: HP
"""
from scipy.stats import norm
import numpy as np
from math import exp, log, sqrt
N = norm.cdf
n = norm.pdf

class Greeks():
    def __init__(self,S0,K,sigma,r,expiry,flag):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.expiry = expiry
        self.flag = flag
        self.discount = exp(-self.r * self.expiry)
        self.d1 = (log(self.S0 / self.K) + (self.r + self.Sigma ** 2 / 2.
                    ) * self.expiry) / (self.Sigma * sqrt(self.expiry))
        self.d2 = self.d1 - self.Sigma * sqrt(self.expiry)


    def get_delta(self):
        if self.flag =='put':
            delta = N(self.d1)-1
        else:
            delta = N(self.d1)
        
        return delta
    def get_gamma(self):
        gamma = n(self.d1)/(self.S0 * self.sigma *sqrt(expiry))
        return gamma
    
    def get_theta(self):        
        if self.flag == 'call':
            theta = (-self.r * self.K * self.discount * N(self.d2) 
                     - self.K * self.discount * n(self.d2) * self.Sigma / (2 * sqrt(self.expiry)))
        if self.flag == 'put':
            theta = (self.r * self.K * self.discount * N(-self.d2) 
                     - self.S0 * n(self.d1) * self.Sigma / (2 * sqrt(self.expiry)))
        return theta
    
    def get_vega(self):
        vega = self.S0 * n(self.d1) * sqrt(self.expiry)
        return vega
    
    def get_rho(self):
        if self.flag == 'call':
            rho = self.expiry * self.K * self.discount * N(self.d2)
        else:
            rho = -self.expiry * self.K * self.discount * N(-self.d2)
        return rho



