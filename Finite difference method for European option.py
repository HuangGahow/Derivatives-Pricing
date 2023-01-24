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
        1.Implict: backward difference
            (We could solve linear system by spsolve, splu or Thomas method,
             here we use splu)
        2.Explict: forward difference
        3.Crank Nicolson: 

        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.cp_flag = cp_flag
        self.omega = 1 if cp_flag =='call' else -1 # omega为payoff乘数
        
    def Implict_FD(self, Nstep = 5000, Ntime = 4000):
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.minimum(self.S0, self.K)
        
        x0 = np.log(self.S0)
        x_max = np.log(S_max)  # 将PDE变量由S改为对数收益率x = ln(S)，令对角矩阵值都为常数 
        x_min = np.log(S_min)  
        
        x_vec, dx = np.linspace(x_min, x_max, num = self.Nstep, retstep = True) #空间离散化
        t_vec, dt = np.linspace(0, T, num = self.Ntime, retstep = True) #时间离散化
        
        self.V = np.zeros((self.Nstep, self.Ntime)) #网格构建
        offset = np.zeros(self.Nstep-2) #边界条件
        
        
        self.V[0, :] = np.maximum(self.omega * (S_min - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smin时，必行权
        self.V[-1, :] = np.maximum(self.omega * (S_max - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = np.maximum(self.omega * (np.exp(x_vec) - self.K), 0) #终值条件,t = T
        
        sigma2 = self.sigma ** 2
        dxx = dx **2
        
        #coefficient of matrix
        a = ( (dt/2) * ( (self.r-0.5*sigma2)/dx - sigma2/dxx ) )
        b = ( 1 + dt * ( sigma2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        
        #tri-diagnal matrix
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(self.Nstep-2, self.Nstep-2)).tocsc()

        # Backward iteration
        # 可以考虑用Thomas算法与spsolve,经比较splu应该是最快的
        DD = splu(self.D)
        for i in range(self.Ntime-2,-1,-1):
            offset[0] = a * self.V[0,i]
            offset[-1] = c * self.V[-1,i]; 
            self.V[1:-1,i] = DD.solve(self.V[1:-1,i+1] - offset) 
            
        return np.interp(x0, x_vec, self.V[:,0])  #线性差值取出所求期权价格
    
    
    def Explict_FD(self, Nstep = 300, Ntime = 200):
        self.Nstep = Nstep
        self.Ntime = Ntime
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.minimum(self.S0, self.K)
        
        x0 = np.log(self.S0)
        x_max = np.log(S_max)  # 将PDE变量由S改为对数收益率x = ln(S)，令对角矩阵值都为常数 
        x_min = np.log(S_min)  
        
        x_vec, dx = np.linspace(x_min, x_max, num = self.Nstep, retstep = True) #空间离散化
        t_vec, dt = np.linspace(0, T, num = self.Ntime, retstep = True) #时间离散化
        
        self.V = np.zeros((self.Nstep, self.Ntime)) #网格构建
        offset = np.zeros(self.Nstep - 2) #边界条件
        
        self.V[0, :] = np.maximum(self.omega * (S_min - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smin时，必行权
        self.V[-1, :] = np.maximum(self.omega * (S_max - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = np.maximum(self.omega * (np.exp(x_vec) - self.K), 0) #终值条件,t = T
        
        sigma2 = self.sigma ** 2
        dxx = dx **2
        
        #coefficient of matrix
        a = ( (dt/2) * ( sigma2/dxx - (self.r-0.5*sigma2)/dx) ) #用backward差分推导即可得
        b = ( 1 - dt * ( sigma2/dxx + self.r ) )
        c = ((dt/2) * ( (self.r-0.5*sigma2)/dx + sigma2/dxx ) )
        """
        仔细观察可发现显式和隐式系数有相同部分，可以进行提前分解，共用因子减少计算量
        三对角矩阵也可分解为A+I
        """
        self.D = sparse.diags([a, b, c], [-1, 0, 1], shape=(self.Nstep-2, self.Nstep-2)).tocsc()
        
        for i in range(self.Ntime-2, -1, -1):
            offset[0] = a * self.V[0, i+1]
            offset[-1] = c * self.V[-1,i+1]
            self.V[1:-1, i] = self.D * self.V[1:-1, i+1] + offset
        
        if np.interp(x0, x_vec, self.V[:,0]) > 10000:
            return np.nan
        else:    
            return np.interp(x0, x_vec, self.V[:,0])
        
    
    def CrankNicolson_FD(self, Nstep = 5000, Ntime = 4000, theta = 0.5):
        self.Nstep = Nstep
        self.Ntime = Ntime
        self.theta = theta
        
        S_max = 3 * np.maximum(self.S0, self.K)
        S_min = 1/3 * np.minimum(self.S0, self.K)
        
        x0 = np.log(self.S0)
        x_max = np.log(S_max)  
        x_min = np.log(S_min)  
        
        x_vec, dx = np.linspace(x_min, x_max, num = self.Nstep, retstep = True) #空间离散化
        t_vec, dt = np.linspace(0, T, num = self.Ntime, retstep = True) #时间离散化
        
        self.V = np.zeros((self.Nstep, self.Ntime)) #网格构建  
        
        self.V[0, :] = np.maximum(self.omega * (S_min - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smin时，必行权
        self.V[-1, :] = np.maximum(self.omega * (S_max - self.K * np.exp(-self.r * t_vec)), 0) #边界条件：S=Smax时，必不行权          
        self.V[:,-1] = np.maximum(self.omega * (np.exp(x_vec) - self.K), 0) #终值条件,t = T
        
        sigma2 = self.sigma ** 2
        dxx = dx **2
        I = sparse.eye(self.Nstep -2)
        
        l = 0.5 * (sigma2 / dxx - (r - 0.5 * sigma2) / dx)
        c = - (sigma2 / dxx + r)
        u = 0.5 * (sigma2 / dxx + (r - 0.5 * sigma2) / dx)

        A = sparse.diags([l, c, u], [-1, 0, 1], shape =(self.Nstep -2, self.Nstep -2), format ='csc')
        M1 = I + (1 - self.theta) * dt * A
        M2 = I - self.theta * dt * A
        
        MM= splu(M2)    
        for i in range(self.Ntime - 2, -1, -1):
            U = M1 * self.V[1:-1, i+1]
            U[0] += theta*l*dt*self.V[0, i] + (1-theta)*l*dt*self.V[0, i+1] 
            U[-1] +=theta*u*dt*self.V[-1, i] + (1-theta)*u*dt*self.V[-1, i+1] 
            self.V[1:-1, i] = MM.solve(U)
         
        return np.interp(x0, x_vec, self.V[:,0])
# In[2]
r = 0.1; sigma1 = 0.2               
S0 = 100; x0 = np.log(S0)          
K = 100; T = 1  
cp_flag = 'call'

# In[3]
#划分网格越细，越容易不稳定。将离散步长增大反而得到了接近的结果，为什么？
%%time
A = European_option(S0, K, r, T, sigma1, cp_flag)
explict = A.Explict_FD(Nstep=300,Ntime=300) 
implict = A.Implict_FD() 
CN = A.CrankNicolson_FD(theta = 0.5)
print('European option price by explict difference method:', explict)
print('European option price by implict difference method:', implict)
print('European option price by Crank Nicolson difference method:', CN)
