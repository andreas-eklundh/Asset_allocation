import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils as u


def backtest_naive(data,mu_target):
    # Note when specifying data, then also period should be specified. 
    mu = np.mean([data["10YrReturns"],data["Market Return"]],axis=1)
    sigma = np.cov([data["10YrReturns"],data["Market Return"]])
    mu0 = np.mean(data["RF"])
    mu_e = np.mean([data["10YrReturns"] - data["RF"],data["Market Return"]-data["RF"]],axis=1)
    sigma_e = np.cov([data["10YrReturns"] - data["RF"],data["Market Return"] - data["RF"]])
    mu_target_e = mu_target - mu0
    # Retrieve weigths
    w = u.get_weights(mu,sigma, mu_target,mu_e, sigma_e, mu_target_e)
    w_rf =  [1 - i @ np.ones(len(i)) for i in w]
    w_method = ["R_40/60", "R_MV", "R_MVL", "R_RP", "R_RPL"]
    idx = [i for i in range(0,len(w_method))]
    # We do it in returns
    for i,j in zip(idx,w_method):
        data[j] = 1 + (w_rf[i] * data["RF"] + w[i][0] * data["10YrReturns"] + w[i][1] * data["Market Return"])
    # Now, only the first obs has correct weights used. 
    # From now on accounting of increase (decrease) in wealth must be accounted for
    for i,j in zip(idx,w_method):
        for k in range(1,len(data["RF"] )):
            w0, w1, w2 = w_rf[i] / data[j][k-1], w[i][0]/ data.loc[k-1, j], w[i][1] / data.loc[k-1, j]
            data.loc[k, j] = (1 +  w0 * data.loc[k, "RF"] + w1 * data.loc[k, "10YrReturns"] +
                               w2 * data.loc[k, "Market Return"])*data.loc[k-1, j]

    return data, w0, w1, w2 
