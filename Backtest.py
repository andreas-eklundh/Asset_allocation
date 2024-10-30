import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils as u
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize 


def backtest_naive2(ind, mu_target, olay = False):
    # Calculate statistics and retrieve weights
    mu, sigma = get_stats(ind, mu_target)
    w = u.get_weights2(mu, sigma, mu_target)  # Use a function to get weights based on mu and sigma
    w = np.vstack((w, [1, 0, 0]))  # Appending weights for RF as [1, 0, 0]
    w_method = ['R_40/60', 'R_MV', 'R_MVL', 'R_RP', 'R_RPL', 'RF']

    # Initialize Monthly_Return DataFrame to store monthly returns
    Monthly_Return = pd.DataFrame(index=ind.index, columns=w_method)

    # Apply weights to calculate returns for each strategy
    for i, method in enumerate(w_method):
        Monthly_Return[method] = 1 + (w[i][0] * ind['RF'] + w[i][1] * ind['10YrReturns'] + w[i][2] * ind['Market Return'])    
        # just to return somewthing
        olays = np.zeros(len(w))
    # Apply weights to calculate returns for each strategy (find optimal overlay)
    if olay == True:
        # Initialize Monthly_Return DataFrame to store monthly returns
        for i, method in enumerate(w_method[:-1]):
            # Solves for optimal (minimal std) overlay size for strategy i
            res = minimize(fun = u.olay_opt, x0 = 0.25, method = 'trust-constr', 
                            args =(ind,mu_target,i), bounds = [(0,0.5)])
            olays[i] = res.x
            # Get new stats
            mu, sigma,mod_mkt = get_olay_stats(ind, mu_target,olays[i])
            # Calculate new weights and returns - overwrites old weight (calc weight returns many weights)
            w[i]= u.get_weights2(mu, sigma, mu_target)[i]  # Use a function to get weights based on mu and sigma 
            Monthly_Return[method] = 1 + (w[i][0] * ind['RF'] + w[i][1] * ind['10YrReturns'] + w[i][2] * mod_mkt)  
  
    # Convert monthly returns to cumulative returns
    Cumulative_Returns = Monthly_Return.cumprod()

    # Drop unnecessary columns from the original data
    ind = ind.drop(columns=['10YrReturns', 'Market Return'])
    Cumulative_Returns = pd.concat([Cumulative_Returns, ind["Date"]],axis=1)

    # Metrics Calculation
    mean_monthly_returns = Monthly_Return.drop(columns='RF').mean()-1
    metrics_data = Monthly_Return.subtract(Monthly_Return['RF'], axis=0).drop(columns='RF')
    annualized_excess_returns = metrics_data.mean() * 12  # Annualizing excess returns
    volatility = metrics_data.std() * np.sqrt(12)
    sharpe_ratio = annualized_excess_returns / volatility
    skewness = metrics_data.apply(skew)
    excess_kurtosis = metrics_data.apply(lambda x: kurtosis(x, fisher=True))
    
    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        'Mean Monthly Returns': mean_monthly_returns,
        'Annualized Excess Returns': annualized_excess_returns,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Skewness': skewness,
        'Excess Kurtosis': excess_kurtosis
    })
    
    return Cumulative_Returns, w, Monthly_Return, metrics,olays

# FOLLOWS JOSTEIN'S ECONOMETRIC NOTES CHAPTER 1.5.5.
# Essentially - fit initially, use strategy for k periods. Rebalance using available data
# Then follow new strategy etc.  
# Assumes data is ordered.

def get_stats(train, mu_target):
    mu = np.mean([train['RF'],train['10YrReturns'],train['Market Return']],axis=1)
    sigma = np.cov([train['RF'],train['10YrReturns'],train['Market Return']])
    return mu,sigma

def get_olay_stats(ind,mu_target,olay):
    # The new Equity time-series:
    data_ol_cost = ind.copy()
    return_overlay = - ind["BIG LoPRIOR"] + ind["BIG HiPRIOR"]
    # Find Equity return by add return and deducing costs
    data_ol_cost["Market Return"] =  data_ol_cost["Market Return"]+olay * (return_overlay -
                                                                u.manager_fee(return_overlay))
    data_ol_cost = data_ol_cost.drop(columns = ["BIG LoPRIOR", "BIG HiPRIOR"])
    mu, sigma = get_stats(data_ol_cost, mu_target)
    return mu, sigma, data_ol_cost["Market Return"]

def backtest_k(ind,mu_target,m,l,K,olay = False):
    weights = []
    olays_list = []
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
        # Get stats:
        mu,sigma = get_stats(train, mu_target)
        # Get training weights.
        w = u.get_weights2(mu,sigma, mu_target)
        # functionality for optimal overlay.
        olays = np.zeros(len(w_method)) # set zero if not specified.
        # THIS FUNTIONAILTY IS NOT USED
        if olay == True:
            # For each strategy, solve for overlay and find optimal weights given history
            olays=np.array([0.17601239,0.17844658,0.14693794,0.06938174,0.13031504,0])
#            for i in range(0,len(w)):
#                res = minimize(fun = u.olay_opt, x0 = 0.25, method = 'trust-constr', 
#                                args =(train,mu_target,i), bounds = [(0,0.5)])
#                olays[i] = res.x
#                mu, sigma,mod_mkt = get_olay_stats(train, mu_target,olays[i])
                # Aquire optimal weights under strategy i's overlay.
#                w[i] = u.get_weights2(mu, sigma, mu_target)[i] 
        # Functionality to keep weights - works for new weights too
        weights.append(w)
        # Append current overlay set
        olays_list.append(olays) 
        w = np.array(w).reshape((len(w),len(mu))) # slight format change
        w = np.vstack((w, new_row))
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
        # Functionality for overlay again.
        add_to_mkt = np.zeros(len(preds)) # is ignored of no overlay
        if olay == True:
            # Finds overlay return (but no wright imposed)
            return_overlay = - preds["BIG LoPRIOR"] + preds["BIG HiPRIOR"]
            add_to_mkt =  np.array(return_overlay - u.manager_fee(return_overlay))


        # Get return of each strategy
        for k in range(0,K_prime):
            if (k == 0):
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return = (1 +  w0 * pred_ret[k,0] + 
                                w1 * pred_ret[k,1] +
                                w2 * (pred_ret[k,2] + olays*add_to_mkt[k])) 
                Monthly_Return[t + k, :] = period_return  # Store monthly returns
                pred[k, :] = period_return*acc_return
            else:
                km1 = k - 1
                w0, w1, w2 =  w[:,0], w[:,1], w[:,2]
                period_return =  (1 + w0 * pred_ret[k,0] + w1 * pred_ret[k,1] +
                                    w2 * (pred_ret[k,2]+ olays*add_to_mkt[k]))
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
    mean_monthly_returns = monthly_out.drop(columns='RF').mean()-1
    metrics_data = monthly_out.iloc[m:, :-1].subtract(monthly_out['RF'].iloc[m:], axis=0)
    annualized_excess_returns = metrics_data.mean() * 12  # Annualizing excess returns
    volatility = metrics_data.std() * np.sqrt(12)
    sharpe_ratio = annualized_excess_returns / volatility  # Correctly calculate Sharpe ratio using annualized data
    skewness = metrics_data.apply(skew)
    excess_kurtosis = metrics_data.apply(lambda x: kurtosis(x, fisher=True))

    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        'Mean Monthly Returns': mean_monthly_returns,
        'Annualized Excess Returns': annualized_excess_returns,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Skewness': skewness,
        'Excess Kurtosis': excess_kurtosis
    })

    return out, weights, monthly_out, metrics, olays_list