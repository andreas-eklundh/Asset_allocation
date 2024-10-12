import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def PV(C,Y,T):
    pv = 0
    if T == 10:
        t_j = [l for l in range(1,10)] 
    else:
        t_j = [l-1/12 for l in range(1,10)]

    for i in t_j:
        pv +=  C / (1+Y)**(i) 
    pv += C / (1+Y)**(T) +   1 / (1+Y)**T 
    return pv

def get_r(Yim1, Yi, Tm1, T):
    pv1 = PV(Yim1,Yi,Tm1)
    pv2 = PV(Yim1,Yim1,T)

    r = pv1 / pv2 - 1

    return r
    

# Functionality for finding minimum variance PF traditionally.

def min_var(mu,sigma, mu_target):
    sigma_inv = np.linalg.inv(sigma)
    o = np.ones(len(mu))
    a = mu @ sigma_inv @ mu
    b = mu @ sigma_inv @ o
    c = o  @ sigma_inv @ o
    A = np.array([[a, b],
                  [b,c]])
    A_inv = np.linalg.inv(A)
    w_target = sigma_inv @ np.array([mu,o]).T @ A_inv @ np.array([mu_target, 1])

    std = np.sqrt(w_target @ sigma @ w_target)

    return w_target, std

# Functionality for finding minimum variance PF with risk free asset. Leverage allowed.
def min_var_rf(mu,sigma, mu_target):
    '''
    Note: Input exess return.
    '''
    sigma_inv = np.linalg.inv(sigma)
    o = np.ones(len(mu))
    w_target = mu_target * sigma_inv @ mu / (mu.T @ sigma_inv @ mu)

    std = np.sqrt(w_target @ sigma @ w_target)

    w0 = 1 - w_target @ o 

    return w0, w_target, std

def risk_parity_fun(w,sigma):
    std = np.sqrt(w @ sigma @ w)
    N  = len(w)
    w_rp = std**2 / ((sigma @ w) * N)
    fun = np.sum((w - w_rp)**2)
    
    return fun

