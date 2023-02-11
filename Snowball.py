# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:42:29 2023

@author: HP
"""
import numpy as np
import pandas as pd
import scipy.stats as ss

# In[1]
class Snowball:
    """
    parameters
    ====================
    S0: spot price of stock
    coupon_rate: coupon rate
    r: risk free rate
    T: time to maturity
    bound: knock in and knock out level
    model: BSM & Heston
    nominal_amount: size of snowball
    N_path: simulate stock prices process paths
    N_time: discrete time number
    start_date: start date of the snow ball option
    check_knockout_date: check knock out dates
    
    
    
    ====================
    
    returns
    ====================
    price: simulated by mc
    price_std: standard deviation of simulation price
    ====================
    """
    T = 2
    nominal_amount = 1000000
    coupon_rate = 0.152
    r = 0.0273  # China 10y Government Bond annual yield
    bound = [0.75, 1.0]
    model = BSM_model
    N_path = 50000
    N_step = T * 365
    dt = 1/365
    
    def __init__(self, S0, coupon_rate, r, T, bound, N_path):
        self.S0 = S0
        self.coupon_rate = coupon_rate
        self.r = r
        self.T = T
        self.knock_in = bound[0]
        self.knock_out = bound[1]
        self.model = model
        self.N_path = N_path
   
    @staticmethod
    def get_price_MC(self, 
                     S0, 
                     coupon_rate, 
                     r, 
                     T, 
                     sigma, 
                     knock_in, 
                     knock_out, 
                     N_path):

        #Simulate the stock price paths in risk-neutral world
        dt = 1/365
        N_step = T * 365
        X_0 = np.zeros((N_path, 1))
        increments = np.random.normal(loc=(r - sigma ** 2 /2)*dt, scale = np.sqrt(dt) * sigma,
                                      size = (N_paths, N_step-1))
        X = np.concatenate((X_0, increments), axis=1).cumsum(1)
        S = S0 * exp(X)
        
        
        df_res = pd.DataFrame(index = range(N_path), columns = ['knock_out', 'knock_in', 'expire date', 'discounted payoff'])
        """Modified: check date could be a input"""
        check_date = [i for i in range(90, T * 365, 30)]
        bool_knock_out = (S[:, check_date] > knock_out*S0)
        whether_knockout = np.sum(bool_knock_out,axis=1)
        df_res.loc[whether_knockout == 0,'knock_out'] = False # path has not knocked out
        df_res.loc[whether_knockout !=0,'knock_out'] = True # path has knocked out

        # Knock out path
        knockout_id = df_res[df_res['knock_out'] == True].index
        for i in knockout_id:
            first_knockout_date = check_date[np.where(bool_knock_out[i,:] == True)[0][0]]
            df_res.loc[i,'expire date'] = first_knockout_date
            df_res.loc[i,'stock price'] = S[i, first_knockout_date]
              
   
        df_res['discounted payoff'] = (1 + df_res['expire date']/365 * coupon_rate) *  np.exp((-r * df_res['expire date']/365).astype(float))
    
        # Knock in path
        bool_knock_in = (S < knock_out * S0)
        whether_knockin = np.sum(bool_knock_in,axis=1)
        df_res.loc[whether_knockin == 0,'knock_in'] = False # path has not knocked in
        df_res.loc[whether_knockin != 0,'knock_in'] = True # path has knocked in
        
        # No knock in & no knock out path
        df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'expire date'] = check_date[-1] 
        df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'stock price'] = S[(whether_knockin == 0) & (whether_knockout == 0), -1]
        df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'discounted payoff'] = (1 +check_date[-1]/365 *coupon_rate) *  np.exp((-r * check_date[-1]/365))
        
        # Knock in but final price fall in [spot price, knock out]
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'expire date'] = check_date[-1] # no knock in & no knock out path
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'stock price'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), -1]
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'discounted payoff'] = np.exp((-r * check_date[-1]/365))
        
        # Knock in but final price is less than spot price
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'expire date'] = check_date[-1]
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'stock price'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), -1]
        df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'discounted payoff'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), -1]/ S0 * np.exp((-r * check_date[-1]/365))
        
        return df_res['discounted payoff'].mean(), df_res['discounted payoff'].std()   
                
# In[1]
"""test
================================
N_path = 300
N_step = 365*2
r = 0.0273
sigma = 0.2225
dt = 1/365
S0 = 7118 
knock_out = 1.0
knock_in = 0.75
T=2
nominal_amount = 1000000
coupon_rate = 0.152


df_res = pd.DataFrame(index = range(N_path), columns = ['knock_out', 'knock_in', 'expire date','stock price', 'discounted payoff'])


X_0 = np.zeros((N_path, 1))
increments = np.random.normal(loc=(r - sigma ** 2 /2)*dt, scale = np.sqrt(dt) * sigma, size = (N_path, N_step-1))
X = np.concatenate((X_0, increments), axis=1).cumsum(1)
S = S0 * np.exp(X)

# In[3]
check_date = [i for i in range(90, T * 365, 30)]
#check_knock_out_price = S[:, check_date]
bool_knock_out = (S[:, check_date] > knock_out*S0)
whether_knockout = np.sum(bool_knock_out,axis=1)
df_res.loc[whether_knockout == 0,'knock_out'] = False # path has not knocked out
df_res.loc[whether_knockout !=0,'knock_out'] = True # path has knocked out

# Knock out path
knockout_id = df_res[df_res['knock_out'] == True].index
for i in knockout_id:
   first_knockout_date = check_date[np.where(bool_knock_out[i,:] == True)[0][0]]
   df_res.loc[i,'expire date'] = first_knockout_date
   df_res.loc[i,'stock price'] = S[i, first_knockout_date]
   
df_res['discounted payoff'] = (1 + df_res['expire date']/365 * coupon_rate) *  np.exp((-r * df_res['expire date']/365).astype(float))

# knock in path
bool_knock_in = (S < knock_out * S0)
whether_knockin = np.sum(bool_knock_in,axis=1)
df_res.loc[whether_knockin == 0,'knock_in'] = False # path has not knocked in
df_res.loc[whether_knockin != 0,'knock_in'] = True # path has knocked in

# No knock in & no knock out path
df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'expire date'] = check_date[-1] 
df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'stock price'] = S[(whether_knockin == 0) & (whether_knockout == 0), -1]
df_res.loc[(whether_knockin == 0) & (whether_knockout == 0), 'discounted payoff'] = (1 +check_date[-1]/365 *coupon_rate) *  np.exp((-r * check_date[-1]/365))

# Knock in but final price fall in [spot price, knock out]
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'expire date'] = check_date[-1] # no knock in & no knock out path
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'stock price'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), -1]
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] > S0), 'discounted payoff'] = np.exp((-r * check_date[-1]/365))

# Knock in but final price is less than spot price
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'expire date'] = check_date[-1]
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'stock price'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), -1]
df_res.loc[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), 'discounted payoff'] = S[(whether_knockin != 0) & (whether_knockout == 0) & (S[:,-1] < S0), -1]/ S0 * np.exp((-r * check_date[-1]/365))


print(df_res['discounted payoff'].mean()*nominal_amount)
df_res['discounted payoff'].std()
=======================================
"""