import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils as u


def backtest_naive(ind,mu_target):
    # Note when specifying ind, then also period should be specified. 
    mu = np.mean([ind['10YrReturns'],ind['Market Return']],axis=1)
    sigma = np.cov([ind['10YrReturns'],ind['Market Return']])
    mu0 = np.mean(ind['RF'])
    mu_e = np.mean([ind['10YrReturns'] - ind['RF'],ind['Market Return']-ind['RF']],axis=1)
    sigma_e = np.cov([ind['10YrReturns'] - ind['RF'],ind['Market Return'] - ind['RF']])
    mu_target_e = mu_target - mu0
    # Retrieve weigths
    w = u.get_weights(mu,sigma, mu_target,mu_e, sigma_e, mu_target_e)
    w_rf =  [1 - i @ np.ones(len(i)) for i in w]
    w_method = ['R_40/60', 'R_MV', 'R_MVL', 'R_RP', 'R_RPL']
    idx = [i for i in range(0,len(w_method))]
    out = ind.copy()
    # We do it in returns
    for i,j in zip(idx,w_method):
        out[j] = 1 + (w_rf[i] * out['RF'] + w[i][0] * out['10YrReturns'] + w[i][1] * out['Market Return'])
    # Now, only the first obs has correct weights used. 
    # From now on accounting of increase (decrease) in wealth must be accounted for
    for i,j in zip(idx,w_method):
        for k in range(1,len(out['RF'] )):
            km1 = k - 1
            w0, w1, w2 = w_rf[i] / out.loc[km1, j], w[i][0]/ out.loc[km1, j], w[i][1] / out.loc[km1, j]
            out.loc[k, j] = (1 +  w0 * out.loc[k, 'RF'] + w1 * out.loc[k, '10YrReturns'] +
                               w2 * out.loc[k, 'Market Return'])*out.loc[km1, j]

    return out, w0, w1, w2 
