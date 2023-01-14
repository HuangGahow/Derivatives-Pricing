# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:38:46 2023

@author: HP
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import splu
import time
        
 # In[1]
class European_option():
    def __init__(self, S0, K, r, T, sigma, cp_flag):
        """
        
        
        Parameters
        ----------
        S : stock price
        K : excercise price
        r : risk free rate
        T : time of expiry
        sigma : volatility
        cp_flag : put or call
        
        Returns
        -------
        European option's price by finite difference method
        1.Implict
        2.Explict
        3.Crank Nicolson

        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.cp_flag = cp_flag
        self.omega = 1 if cp_flag =='call' else 0 # omega为payoff乘数
        
    def Implict_PD(self, Nstep = 3000, Ntime = 2000):
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.minimum(self.S0, self.K)
        
        x0 = np.log(S0)
        x_max = np.log(S_max)  # 将PDE变量由S改为对数收益率x = ln(S)，令对角矩阵值都为常数 
        x_min = np.log(S_min)  
        
        x_vec, dx = np.linspace(x_min, x_max, num = Nstep, retstep = True) #空间离散化
        T_vec, dt = np.linspace(0, T, num = Ntime, retstep = True) #时间离散化
        
        self.V = np.zeros((Nstep, Ntime)) #网格构建
        offset = np.zeros(Nstep-2) #边界条件
        self.V[0, :] = np.maximum(self.omega * (S_min - self.K * np.exp(-self.r * T_vec)), 0) #边界条件：S=Smin时，必行权
        self.V[-1, :] = np.maximum(self.omega * (S_max - self.K * np.exp(-self.r * T_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = np.maximum(self.omega * (np.exp(x_vec) - self.K), 0) #终值条件,t = T
        
        sigma2 = sigma ** 2
        dxx = dx **2
        
        #coefficient of matrix
        a = ( (dt/2) * ( (self.r-0.5*sigma2)/dx - sigma2/dxx ) )
        b = ( 1 + dt * ( sigma2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        
        #tri-diagnal matrix
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nstep-2, Nstep-2)).tocsc()

        # Backward iteration
        # 可以考虑用Thomas算法与spsolve,经比较splu应该是最快的
        DD = splu(self.D)
        for i in range(Ntime-2,-1,-1):
            offset[0] = a * self.V[0,i]
            offset[-1] = c * self.V[-1,i]; 
            self.V[1:-1,i] = DD.solve(self.V[1:-1,i+1] - offset) 
            
        return np.interp(x0, x_vec, self.V[:,0])  #线性插值取出所求期权价格


  def Explict_FD(self, Nstep = 30, Ntime = 20):
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.minimum(self.S0, self.K)
        
        x0 = np.log(S0)
        x_max = np.log(S_max)  # 将PDE变量由S改为对数收益率x = ln(S)，令对角矩阵值都为常数 
        x_min = np.log(S_min)  
        
        x_vec, dx = np.linspace(x_min, x_max, num = Nstep, retstep = True) #空间离散化
        T_vec, dt = np.linspace(0, T, num = Ntime, retstep = True) #时间离散化
        
        self.V = np.zeros((Nstep, Ntime)) #网格构建
        offset = np.zeros(Nstep-2) #边界条件
        
        self.V[0, :] = np.maximum(self.omega * (S_min - self.K * np.exp(-self.r * T_vec)), 0) #边界条件：S=Smin时，必行权
        self.V[-1, :] = np.maximum(self.omega * (S_max - self.K * np.exp(-self.r * T_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = np.maximum(self.omega * (np.exp(x_vec) - self.K), 0) #终值条件,t = T
        
        sigma2 = sigma ** 2
        dxx = dx **2
        
        #coefficient of matrix
        a = ( (dt/2) * ( sigma2/dxx - (self.r-0.5*sigma2)/dx) ) #用backward差分推导即可得
        b = ( 1 - dt * ( sigma2/dxx + self.r ) )
        c = ((dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        """
        仔细观察可发现显式和隐式系数有相同部分，可以进行提前分解，共用因子减少计算量
        三对角矩阵也可分解为A+I
        """
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nstep-2, Nstep-2)).tocsc()
        
        for i in range(Ntime-2, -1, -1):
            offset[0] = a * self.V[0, i+1]
            offset[-1] = c * self.V[-1,i+1]
            self.V[1:-1, i] = self.D * self.V[1:-1, i+1] + offset
        
        if np.interp(x0, x_vec, self.V[:,0]) == np.nan:
            print('no convergent solution ')
        else:    
            return np.interp(x0, x_vec, self.V[:,0])

# In[2]
r = 0.1; sigma = 0.2               
S0 = 100; x0 = np.log(S0)          
K = 100; T = 1  
cp_flag = 'call'

# In[3]
A = European_option(S0, K, r, T, sigma, cp_flag)
explict = A.Explict_FD() #划分网格越细，越容易不稳定。将离散步长增大反而得到了接近的结果，为什么？
implict = A.Implict_FD() 
