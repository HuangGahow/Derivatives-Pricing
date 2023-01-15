# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:24:18 2022

@author: HP
"""

import numpy as np
S0 = 100
K = 100
T = 1
r = 0.1
sig = 0.2
payoff = "put"
N=500
def getAmeOption(S0, K, T, r, sig, N, payoff):
    # 将时间T拆分为N份，每份是dT。
    dT = float(T) / N
    #计算u，d，p，q。
    u = np.exp(sig * np.sqrt(dT))
    d = 1.0 / u   
    a = np.exp(r * dT)
    p = (a - d)/(u - d)
    q = 1.0 - p    
    # V承接N+1个价格
    V = np.zeros(N+1)
    # 注意S_T的生成顺序，是从u=0,d=N开始.                            
    S_T = np.array( [(S0 * u**j * d**(N - j)) for j in range(N + 1)] ) # price S_T at time 
    
    if payoff =="call": 
        V[:] = np.maximum(S_T-K, 0.0)
    elif payoff =="put":
        V[:] = np.maximum(K-S_T, 0.0)

    for i in range(N-1, -1, -1):
        # 这一步至关重要：V在第k轮迭代中，只关注0~N-k的位置，
        # 每个位置=（下一个位置上升的期权*p+下一个位置下降的期权*q）*折现因子
        V[:-1] = np.exp(-r*dT) * (p * V[1:] + q * V[:-1])   
        #截头跟截尾相加
        # 股价也进行新一轮的迭代，同样只有0~N-k是需要关注的，剩下的位置无关紧要。
        S_T = S_T * u       
        # 比较此刻行权和下一轮预期收益的折现
        if payoff=="call":
            V = np.maximum( V, S_T-K )
        elif payoff=="put":
            V = np.maximum( V, K-S_T )
            
    return V[0]
        
getAmeOption(S0, K, T, r, sig, 10000, 'put')
getAmeOption(S0, K, T, r, sig, 10000, 'call')



from random import gauss
from math import exp, sqrt

def calculate_S_T(S, v, r, T):
    """模拟epsilon，计算S_T"""
    return S * exp((r - 0.5 * v ** 2) * T + v * sqrt(T) * gauss(0.0, 1.0))

def getRoundStockPrice(S, v, r, N, T, rounds, paths):
    firstRound = [S] * paths    
    nextRound = firstRound.copy()
    for i in range(rounds):
        nextRound1 = nextRound[-paths:]        
        for s in nextRound1:
            nextS = calculate_S_T(s, v, r, T)
            nextRound.append(nextS)
    nextRound = np.array(nextRound).reshape((-1, paths)).T #reshape按行填充
    return nextRound

S = 100
v = 0.2
r = 0.1
N = 100
T = 1 / N
paths = 100
S_Matrix = getRoundStockPrice(S, v, r, N, T, N, paths)



import scipy.stats as ss
import numpy as np

def LSM(S0, K, T, r, sig, payoff, N=10000, paths=10000, order=2):

    dt = T/(N-1)          # time interval
    df = np.exp(r * dt)  # discount factor per time time interval

    X0 = np.zeros((paths,1))
    increments = ss.norm.rvs(loc=(r - sig**2/2)*dt, scale=np.sqrt(dt)*sig, size=(paths,N-1))
    X = np.concatenate((X0,increments), axis=1).cumsum(1) #cumsum内是按轴1cumsum
    S = S0 * np.exp(X)
    if payoff == "put":
        H = np.maximum(K - S, 0)   # intrinsic values for put option exercise value
    if payoff == "call":
        H = np.maximum(S - K, 0)   # intrinsic values for call option exercise value
    V = np.zeros_like(H)            # value matrix
    V[:,-1] = H[:,-1]

    # Valuation by LS Method
    for t in range(N-2, 0, -1):
        good_paths = H[:,t] > 0    
        # polynomial regression：将EV>0的部分挑出来回归
        rg = np.polyfit( S[good_paths, t], V[good_paths, t+1] * df, 2)  #S[good_paths, t]通过good_path bool 提取  
        # 估计E(HV)
        C = np.polyval( rg, S[good_paths,t] )                             
        # 如果E(HV)<EV，那么行权
        exercise = np.zeros( len(good_paths), dtype=bool)
        exercise[good_paths] = H[good_paths,t] > C #行权！
        V[exercise,t] = H[exercise,t]
        V[exercise,t+1:] = 0
        discount_path = (V[:,t] == 0) #V[:t]==0的部分即holding value大于exercise value，价值等于上一步折现
        V[discount_path,t] = V[discount_path,t+1] * df
    V0 = np.mean(V[:,1]) * df  # 
    return V0
call = LSM(100, 100, 1, 0.1, 0.2, 'call', N=10000, paths=10000, order=2)
put = LSM(100, 100, 1, 0.1, 0.2, 'put', N=10000, paths=10000, order=2)

t=98
        good_paths = H[:,t] > 0    
        # polynomial regression：将EV>0的部分挑出来回归
        rg = np.polyfit( S[good_paths, t], V[good_paths, t+1] * df, 2)    
        # 估计E(HV)
        C = np.polyval( rg, S[good_paths,t] )                             
        # 如果E(HV)<EV，那么行权
        exercise = np.zeros( len(good_paths), dtype=bool)
        exercise[good_paths] = H[good_paths,t] > C
        V[exercise,t] = H[exercise,t]
        V[exercise,t+1:] = 0
        discount_path = (V[:,t] == 0)
        V[discount_path,t] = V[discount_path,t+1] * df
    V0 = np.mean(V[:,1]) * df 
S[good_paths, t]
