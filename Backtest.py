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

# FOLLOWS JOSTEIN'S ECONOMETRIC NOTES CHAPTER 1.5.5.
# Essentially - fit initially, use strategy for k periods. Rebalance using available data
# Then follow new strategy etc.  
# Assumes data is ordered.

def get_stats(train, mu_target):
    mu = np.mean([train['10YrReturns'],train['Market Return']],axis=1)
    sigma = np.cov([train['10YrReturns'],train['Market Return']])
    mu0 = np.mean(train['RF'])
    mu_e = np.mean([train['10YrReturns'] - train['RF'],train['Market Return']-
                        train['RF']],axis=1)
    sigma_e = np.cov([train['10YrReturns'] - train['RF'],train['Market Return'] - 
                         train['RF']])
    mu_target_e = mu_target - mu0
    return mu,sigma ,mu0,mu_e,sigma_e,mu_target_e


# Note as such that this method primarly makes sense with k=l i.e. non overlapping prediction 
def backtest_k(ind,mu_target,m,l,K):
    w_method = ['R_40/60', 'R_MV', 'R_MVL', 'R_RP', 'R_RPL']
    n = len(ind)
    # Output DF of length n-m
    acc_return = np.ones(5) # 5 strategies.
    train_incr = [m+r*l for r in range(0,int(np.floor((n - K -m) / l)))]
    Return = np.full(shape=(len(ind),5),fill_value=1.0)
    for t in train_incr:
        K_prime = K
        train = ind.loc[0:t-1] # get first t obs
        preds = ind.loc[t:t+K-1]
        # validat en K obs
        # Get returns for calculations:
        pred_ret = np.array(preds[['RF','10YrReturns','Market Return']] )
        pred = np.ones(shape=(K,5)) # Pd accounts for both sides thus -1 
        # To get full prediction in case of the final test
        if t == train_incr[-1]:
            preds = ind.loc[t:]
            # Get returns for calculations:
            pred_ret = np.array(preds[['RF','10YrReturns','Market Return']] )
            pred = np.ones(shape=(len(preds),len(w_method))) # Pd accounts for both sides thus -1 
            K_prime = len(preds)

        mu,sigma,mu0,mu_e,sigma_e,mu_target_e = get_stats(train, mu_target)
        # Get training weights.
        w = u.get_weights(mu,sigma, mu_target,mu_e,sigma_e,mu_target_e)
        w = np.array(w).reshape((len(w),len(mu))) # slight format change
        # Get Risk free (might delete later.)
        w_rf =  np.array([1 - i @ np.ones(len(i)) for i in w])
        # Get return of each strategy
        for k in range(0,K_prime):
            if (k == 0):
                w0, w1, w2 = w_rf / acc_return, w[:,0]/ acc_return, w[:,1] / acc_return
                pred[k, :] = acc_return*(1 +  w0 * pred_ret[k,0] + 
                                                    w[:,0] * pred_ret[k,1] +
                                                    w[:,1]* pred_ret[k,2])
            else:
                km1 = k - 1
                w0, w1, w2 = w_rf / pred[km1, :], w[:,0]/ pred[km1, :], w[:,1] / pred[km1, :]
                pred[k, :] = (1 + w0 * pred_ret[k,0] + w1 * pred_ret[k,1] +
                                                    w2 * pred_ret[k,2])* pred[km1, :]
        acc_return =  pred[-1, :]
        # Out 
        if t == train_incr[-1]:
            Return[t:] = pred
        else:
            Return[t:t+K] = pred

 
    out = pd.DataFrame(columns = w_method, data=Return)
    out = pd.concat([out, ind["Date"]],axis=1)
    return out
