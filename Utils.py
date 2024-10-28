import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize 
import cvxpy as cp

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

def manager_fee(r):
    return 15 / 10000 + 0.1 * np.max([(r - 100 / 10000),np.zeros(len(r))],axis=0)
    

# Functionality for finding minimum variance PF traditionally.

def mv_analysis(mu,sigma,mu_target):
    sigma_inv = np.linalg.inv(sigma)
    o = np.ones(len(mu))
    a = mu.T @ sigma_inv @ mu
    b = mu.T @ sigma_inv @ o
    c = o.T  @ sigma_inv @ o
    A = np.array([[a, b],
                  [b,c]])
    A_inv = np.linalg.inv(A)
    w_target = sigma_inv @ np.array([mu,o]).T @ A_inv @ np.array([mu_target, 1])

    std = np.sqrt(w_target @ sigma @ w_target)

    return w_target, std

def get_maxret(mu,sigma,mu_target,constr):
    n = mu.shape[0]
    # get mv stats
    w = cp.Variable(n)
    portfolio_return = mu.T @ w
    # We must find a minimum return first:
    constraints = [cp.sum(w) == 1,      # Sum of weights must be 1
    w[0] >= constr,                          # No short-selling for asset Money
    w[1] >= 0,                          # No short-selling for asset Bonds
    w[2] >= 0]
        
    objective = cp.Maximize(portfolio_return)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = w.value
    # Note these weights should be optimal - or suboptimal.
    mu_opt = optimal_weights @ mu
    return mu_opt, optimal_weights

def variance(weights, sigma):
    return 0.5 * weights @ sigma @ weights


def min_var(mu,sigma, mu_target):
    # Solve only for two assets.
    n = mu.shape[0]
    w = cp.Variable(n)
    portfolio_return = mu.T @ w
    portfolio_variance = cp.quad_form(w, sigma)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(w) == 1,        # Sum of weights must be 1
                w[0] >= 0,              # No short-selling for asset Money
                w[1] >= 0,              # No short-selling for asset Bonds
                w[2] >= 0,              # No short-selling for asset Stocks
                portfolio_return >= mu_target]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = w.value
    if optimal_weights is None:
        # Problem is more delicate
        mu_max, weigths = get_maxret(mu, sigma, mu_target, constr=0)
        constraints = [cp.sum(w) == 1,        # Sum of weights must be 1
            w[0] >= 0,                        # No short-selling for asset Money
            w[1] >= 0,              # No short-selling for asset Bonds
            w[2] >= 0,              # No short-selling for asset Stocks
            portfolio_return >= mu_max] 
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_weights = w.value
        # this latter part is only numerical instability - take weights from before.
        if optimal_weights is None:
            optimal_weights = weigths
    std = np.sqrt(optimal_weights @ sigma @ optimal_weights)

    return optimal_weights, std

# Functionality for finding minimum variance PF with risk free asset. Leverage allowed.
def min_var_rf(mu,sigma, mu_target):
    # Solve only for two assets.
    n = mu.shape[0]
    w = cp.Variable(n)
    portfolio_return = mu.T @ w
    portfolio_variance = cp.quad_form(w, sigma)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(w) == 1,        # Sum of weights must be 1
                w[0] >= -0.50,              # No short-selling for asset 1
                w[1] >= 0, 
                w[2] >= 0,             # No short-selling for asset 2
                portfolio_return >= mu_target]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = w.value
    if optimal_weights is None:
        # Problem is more delicate
        mu_max, weights = get_maxret(mu, sigma, mu_target, constr=-0.5)
        constraints = [cp.sum(w) == 1,        # Sum of weights must be 1
            w[0] >= -0.5,           # No short-selling for asset Money
            w[1] >= 0,              # No short-selling for asset Bonds
            w[2] >= 0,              # No short-selling for asset Stocks
            portfolio_return >= mu_max]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_weights = w.value
        # this latter part is only numerical instability - take weights from before.
        if optimal_weights is None:
            optimal_weights = weights
    std = np.sqrt(optimal_weights @ sigma @ optimal_weights)

    return optimal_weights, std


def risk_parity_fun(w,sigma,lev):
    if lev == True:
        w = w[1:]
    std = np.sqrt(w @ sigma @ w)
    N  = len(w)
    w_rp = std**2 / ((sigma @ w) * N)
    fun = np.sum((w - w_rp)**2)
    
    return fun


def risk_parity(mu,sigma,mu_target):
    constraints_rp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Target return

    N = int(len(sigma[1:,0]))
    w0 = np.ones(N) / N
    res = minimize(fun = risk_parity_fun, x0 = w0, method = 'trust-constr', 
                        args =(sigma[1:,1:],False),
                        bounds = ((0,1),(0,1)),
                            constraints=constraints_rp)
    w_rp = res.x
    w_rp = np.append(0,w_rp)
    sigma_rp = np.sqrt(w_rp @ sigma @w_rp)

    return w_rp, sigma_rp

def levered_risk_parity(mu,sigma,mu_target):
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights >= 1
                    {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - mu_target}  
 ]
    N = int(len(sigma[0,:]))
    w0 = np.ones(N) / N
    # Note we impose to actually borrow - or not but no investment. 
    res = minimize(fun = risk_parity_fun, x0 = w0, method = 'trust-constr', 
                        args =(sigma[1:,1:],True),
                        bounds = ((-0.5,0),(0,1.5),(0,1.5)),
                            constraints=constraints)
    w_rp_lev = res.x
    sigma_rp_lev = np.sqrt(w_rp_lev @ sigma @ w_rp_lev)
    
    return w_rp_lev,sigma_rp_lev

def rp_gb(sigma):
    sigma_roll = np.sqrt(np.linalg.diagonal(sigma)[1:])
    k_t = 1/np.sum(1/sigma_roll)
    w_rp = k_t * 1/sigma_roll
    w_rp = np.append(0,w_rp) # get RF weights.

    return w_rp

def rp_gb_lev(w_rp,mu, mu_target):
    l_r = (mu_target - mu[0]) / (w_rp @ mu - mu[0]*(w_rp[1]+ w_rp[2]))
    l_t = np.max([np.min([l_r, 1.5]),1])
    w_rp_lev = l_t * w_rp.copy()
    w0 = 1-  w_rp_lev[1:] @ np.ones(len(w_rp_lev[1:]))
    w_rp_lev[0] = w0 # get RF weights.

    return w_rp_lev

def get_weights(mu,sigma, mu_target):
    w_4060 = np.array([0,0.40,0.60])
    w_MV = min_var(mu,sigma, mu_target)[0]
    w_MVL = min_var_rf(mu,sigma, mu_target)[0]
    w_RP = risk_parity(mu,sigma,mu_target)[0]
    w_RPl = levered_risk_parity(mu,sigma,mu_target)[0]

    return [w_4060,w_MV,w_MVL,w_RP,w_RPl]


def get_weights2(mu,sigma, mu_target):
    w_4060 = np.array([0,0.40,0.60])
    w_MV = min_var(mu,sigma, mu_target)[0]
    w_MVL = min_var_rf(mu,sigma, mu_target)[0]
    w_RP = rp_gb(sigma)
    w_RPl = rp_gb_lev(w_RP,mu, mu_target)

    return [w_4060,w_MV,w_MVL,w_RP,w_RPl]

# Functionality for returning std of olay's
def olay_opt(olay, data,mu_target,type):
    data_ol_cost = data.copy()
    # Find overlay returns
    return_overlay = - data["BIG LoPRIOR"] + data["BIG HiPRIOR"]
    # Find Equity return by add return and deducing costs
    data_ol_cost["Market Return"] =  data_ol_cost["Market Return"]+ olay * (return_overlay -
                                                                             manager_fee(return_overlay))
    data_ol_cost = data_ol_cost.drop(columns = ["BIG LoPRIOR", "BIG HiPRIOR"])

    mu_cost = np.mean([data_ol_cost["RF"],data_ol_cost["10YrReturns"],data_ol_cost["Market Return"]],axis=1)
    sigma_cost = np.cov([data_ol_cost["RF"],data_ol_cost["10YrReturns"],data_ol_cost["Market Return"]])
    # 'Get weights' that give minimum variance
    weigths = get_weights2(mu_cost,sigma_cost,mu_target)
    # Standard deviation:
    std_arr = np.sqrt(weigths[type] @ sigma_cost @ weigths[type]) 
    
    return std_arr

def table_2_lower(df):
    N = len(df.copy())
    name_list = df.columns.tolist()
    table2 = pd.melt(df, value_vars=name_list)
    table2["ME"] = table2["variable"].str.split().str[0]
    table2["PRIOR"] = table2["variable"].str.split().str[1]
    table2 = table2.drop(columns = "variable")
    table2 = table2.groupby(["ME","PRIOR"])["value"].agg(["mean", "std"]).reset_index()
    table2["t-test of mean"] = table2["mean"] / (table2["std"] / np.sqrt(N))

    table2 = table2.pivot(index = "ME", columns = "PRIOR", values = ["mean", "std","t-test of mean"])

    return table2


'''
def min_var(mu,sigma, mu_target):
    # Solve only for two assets.
    mu2 = mu[1:]
    sigma2 = sigma[1:,1:]
    initial_weights = np.ones(len(mu2)) / len(mu2)

    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}, 
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, mu2) - mu_target}  
    )
    bounds = ((0,1),(0,1))
    result = minimize(variance, initial_weights, args=(sigma2), method='trust-constr', 
                      bounds=bounds, constraints=constraints)
    w = result.x
    w0 = 0 
    w = np.append(w0,w)
    std = np.sqrt(w @ sigma @ w)

    return w, std

# Functionality for finding minimum variance PF with risk free asset. Leverage allowed.
def min_var_rf(mu,sigma, mu_target):
    initial_weights = np.ones(len(mu)) / len(mu)

    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}, 
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, mu) - mu_target}  
    )
    bounds = ((-0.5,0),(0,1.5),(0,1.5))
    # Note bounds say specifically that we only leverage i.e. borrow. 
    result = minimize(variance, initial_weights, args=(sigma), method='trust-constr', 
                      bounds=bounds, constraints=constraints)
    w = result.x
    std = np.sqrt(w @ sigma @ w)

    return w, std
    '''