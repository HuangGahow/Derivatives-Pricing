# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:10:04 2022

@author: HP
"""
import numpy as np 
def get_matrix(M, K, delta_S, call =True):
    # call的三个边界条件
    # 生成(M+1)*(M+1)的矩阵
    f_matrx = np.matrix(np.array([0.0]*(M+1)*(M+1)).reshape((M+1, M+1)))
    # 边界条件① S=0的时候，call=0
    f_matrx[:,0] = 0.0
    # 边界条件②：在到期的时候，期权=max(δS*j-K, 0)
    if call ==True:
        for i in range(M + 1):
            f_matrx[M, i] = float(max(delta_S * i - K, 0))
    # 边界条件③：S=S_max的时候，call=S_max-K
            f_matrx[:,M] = float(S_max - K)
            #print("f_matrix shape : ", f_matrx.shape)
    else:
        for i in range(M + 1):
            f_matrx[M, i] = float(max(K- delta_S * i , 0))
            f_matrx[:,M] = float(S_max - K)
            #print("f_matrix shape : ", f_matrx.shape)
            
    return f_matrx


# In[2]
def calculate_coeff(j):
    vj2 = (v * j)**2
    aj = 0.5 * delta_T * (r * j - vj2)
    bj = 1 + delta_T * (vj2 + r)
    cj = -0.5 * delta_T * (r * j + vj2)
    return aj, bj, cj

def get_coeff_matrix(M):
    #计算系数矩阵B
    matrx = np.matrix(np.array([0.0]*(M-1)*(M-1)).reshape((M-1, M-1)))
    a1, b1, c1 = calculate_coeff(1)
    am_1, bm_1, cm_1 = calculate_coeff(M - 1)
    matrx[0,0] = b1
    matrx[0,1] = c1
    matrx[M-2, M-3] = am_1
    matrx[M-2, M-2] = bm_1    
    for i in range(2, M-1):
        a, b, c = calculate_coeff(i)
        matrx[i-1, i-2] = a
        matrx[i-1, i-1] = b
        matrx[i-1, i] = c    
    #print("coeff matrix shape : ",  matrx.shape)
    return matrx

# In[4]
def calculate_f_matrix(flag, M):
    if flag == "call":
        f_matrx = get_matrix(M, K, delta_S,call=True)
    else:
        f_matrx = get_matrix(M, K, delta_S,call=False)
        
    matrx = get_coeff_matrix(M)
    inverse_m = matrx.I
    for i in range(M, 0, -1):
        # 迭代
        Fi = f_matrx[i, 1:M]
        Fi_1 = inverse_m * Fi.reshape((M-1, 1))
        Fi_1 = list(np.array(Fi_1.reshape(1, M-1))[0])
        f_matrx[i-1, 1:M]=Fi_1    
    # 这一步取出S_t在网格中的位置，然后抽出结果，即为在该股价的期权价格。
    i = np.round(S/delta_S, 0)
    return f_matrx[0, int(i)]



# In[3]
M = 5000
S = 276.10
r = 0.16/100
T = 58 / 365
v = 0.407530933
K = 230
S_max = 500
delta_T = T/M
delta_S = S_max/M
call = calculate_f_matrix('call',M)
put = calculate_f_matrix('put', M)