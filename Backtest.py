import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils as u
from scipy.stats import skew, kurtosis


def backtest_naive(ind,mu_target):
    # Note when specifying ind, then also period should be specified. 
    mu,sigma = get_stats(ind, mu_target)
    # Retrieve weigths
    w = u.get_weights(mu,sigma, mu_target)
    w_method = ['R_40/60', 'R_MV', 'R_MVL', 'R_RP', 'R_RPL']
    idx = [i for i in range(0,len(w_method))]
    out = ind.copy()
    # We do it in returns
    for i,j in zip(idx,w_method):
        out[j] = 1 + (w[i][0] * out['RF'] + w[i][1] * out['10YrReturns'] + w[i][2] * out['Market Return'])
    # Now, only the first obs has correct weights used. 
    # From now on accounting of increase (decrease) in wealth must be accounted for
    for i,j in zip(idx,w_method):
        for k in range(1,len(out['RF'] )):
            km1 = k - 1
            w0, w1, w2 = w[i][0] ,w[i][1], w[i][2] 
            out.loc[k, j] = (1 +  w0 * out.loc[k, 'RF'] + w1 * out.loc[k, '10YrReturns'] +
                               w2 * out.loc[k, 'Market Return'])*out.loc[km1, j]

    return out, w0, w1, w2 

# FOLLOWS JOSTEIN'S ECONOMETRIC NOTES CHAPTER 1.5.5.
# Essentially - fit initially, use strategy for k periods. Rebalance using available data
# Then follow new strategy etc.  
# Assumes data is ordered.

def get_stats(train, mu_target):
    mu = np.mean([train['RF'],train['10YrReturns'],train['Market Return']],axis=1)
    sigma = np.cov([train['RF'],train['10YrReturns'],train['Market Return']])
    return mu,sigma


# Note as such that this method primarly makes sense with k=l i.e. non overlapping prediction 
def backtest_2(ind,mu_target,m,l,K):
    weights = []
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

        mu,sigma = get_stats(train, mu_target)
        # Get training weights.
        w = u.get_weights(mu,sigma, mu_target)
        weights.append(w)
        w = np.array(w).reshape((len(w),len(mu))) # slight format change
        # Get return of each strategy
        for k in range(0,K_prime):
            if (k == 0):
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return = (1 +  w0 * pred_ret[k,0] + 
                                w[:,0] * pred_ret[k,1] +
                                w[:,1] * pred_ret[k,2]) 
                pred[k, :] = period_return*acc_return
            else:
                km1 = k - 1
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return =  (1 + w0 * pred_ret[k,0] + w1 * pred_ret[k,1] +
                                    w2 * pred_ret[k,2])
                pred[k, :] = period_return* pred[km1, :]

        acc_return =  pred[-1, :]
        # Out 
        if t == train_incr[-1]:
            Return[t:] = pred
        else:
            Return[t:t+K_prime] = pred

 
    out = pd.DataFrame(columns = w_method, data=Return)
    out = pd.concat([out, ind["Date"]],axis=1)
    return out, weights


def backtest_k(ind,mu_target,m,l,K):
    weights = []
    w_method = ['R_40/60', 'R_MV', 'R_MVL', 'R_RP', 'R_RPL', 'RF']
    n = len(ind)
    new_row = [1, 0, 0]
    # Output DF of length n-m
    acc_return = np.ones(6) # 5 strategies + RF.
    train_incr = [m+r*l for r in range(0,int(np.floor((n - K -m) / l)))]
    Return = np.full(shape=(len(ind),6),fill_value=1.0)
    Monthly_Return = np.full(shape=(len(ind), 6), fill_value=np.nan)  # To store non-accumulated returns   
    for t in train_incr:
        K_prime = K
        train = ind.loc[0:t-1] # get first t obs
        preds = ind.loc[t:t+K-1]
        # validat en K obs
        # Get returns for calculations:
        pred_ret = np.array(preds[['RF','10YrReturns','Market Return']] )
        pred = np.ones(shape=(K,6)) # Pd accounts for both sides thus -1 
        # To get full prediction in case of the final test
        if t == train_incr[-1]:
            preds = ind.loc[t:]
            # Get returns for calculations:
            pred_ret = np.array(preds[['RF','10YrReturns','Market Return']] )
            pred = np.ones(shape=(len(preds),len(w_method))) # Pd accounts for both sides thus -1 
            K_prime = len(preds)

        mu,sigma = get_stats(train, mu_target)
        # Not really a rolling estimate, but similar. 
        # TODO: consider doing this rolling thing differently.
        sigma_roll = np.cov([train['RF'],train['10YrReturns'],train['Market Return']])
        sigma_roll = np.sqrt(np.linalg.diagonal(sigma_roll)[1:])
        # Get training weights.
        w = u.get_weights2(mu,sigma, mu_target)
        weights.append(w)
        w = np.array(w).reshape((len(w),len(mu))) # slight format change
        w = np.vstack((w, new_row))
        # Get return of each strategy
        for k in range(0,K_prime):
            if (k == 0):
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return = (1 +  w0 * pred_ret[k,0] + 
                                w[:,0] * pred_ret[k,1] +
                                w[:,1] * pred_ret[k,2]) 
                Monthly_Return[t + k, :] = period_return  # Store monthly returns
                pred[k, :] = period_return*acc_return
            else:
                km1 = k - 1
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return =  (1 + w0 * pred_ret[k,0] + w1 * pred_ret[k,1] +
                                    w2 * pred_ret[k,2])
                Monthly_Return[t + k, :] = period_return  # Store monthly returns
                pred[k, :] = period_return* pred[km1, :]

        acc_return =  pred[-1, :]
        # Out 
        if t == train_incr[-1]:
            Return[t:] = pred
        else:
            Return[t:t+K_prime] = pred

 
    out = pd.DataFrame(columns = w_method, data=Return)
    monthly_out = pd.DataFrame(columns=w_method, data=Monthly_Return)
    out = pd.concat([out, ind["Date"]],axis=1)


    # Metrics Calculation Starting from 0 + m
    metrics_data = monthly_out.iloc[m:, :-1].subtract(monthly_out['RF'].iloc[m:], axis=0)
    annualized_excess_returns = metrics_data.mean() * 12  # Annualizing excess returns
    volatility = metrics_data.std() * np.sqrt(12)
    sharpe_ratio = annualized_excess_returns / volatility  # Correctly calculate Sharpe ratio using annualized data
    skewness = metrics_data.apply(skew)
    excess_kurtosis = metrics_data.apply(lambda x: kurtosis(x, fisher=True))

    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        'Annualized Excess Returns': annualized_excess_returns,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Skewness': skewness,
        'Excess Kurtosis': excess_kurtosis
    })

    return out, weights, monthly_out, metrics