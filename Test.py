import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from scipy.optimize import minimize 
import Backtest as bt   
import matplotlib.dates as mdates

mu_target = 75 / 10000 # Target
w_current = np.array([0,0.4, 0.6]) # current strategy
bond = pd.read_csv("Data_clean/bond_returns.csv")
stock = pd.read_csv("Data_clean/6_Portfolios_ME_Prior_12_2_returns.csv")
bond, stock =bond[["Date", "10YrReturns"]], stock[["Date", "Market Return"]]
stock["Market Return"] = stock["Market Return"] /100  
data = pd.merge(bond,stock, how='left', on = "Date")
RF = pd.read_csv("Data_clean/FF_cleaned.csv")
data = pd.merge(data.copy(),RF[["Date","RF"]], 'left',on = "Date" )
data["RF"] = data["RF"] /100 # assumed this must hold

MOMdep = pd.read_csv("Data_clean/25_Portfolios_ME_Prior_12_2_returns.csv")
MOMdep = MOMdep[["Date", "BIG LoPRIOR", "BIG HiPRIOR"]]
data = pd.merge(data.copy(),MOMdep, 'left',on = "Date" )
data["BIG LoPRIOR"], data["BIG HiPRIOR"] =data["BIG LoPRIOR"] /100, data["BIG HiPRIOR"] /100


test2,w0t,w1t,w2t,ol = bt.backtest_naive2(ind=data, mu_target=mu_target, olay=True)

time = pd.date_range(data["Date"][0],data["Date"][len(data["Date"]) -1 ], freq = 'ME')
fig, ax = plt.subplots()
ax.plot(time,test2["R_40/60"], label="R of 40/60")
ax.plot(time,test2["R_MV"], label="R of Mean-Var")
ax.plot(time,test2["R_MVL"], label="R of Mean-Var Leveraged")
ax.plot(time,test2["R_RP"], label="R of Risk Parity")
ax.plot(time,test2["R_RPL"], label="R of Risk Parity Leveraged")
ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Set a tick at the start of every year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend()
plt.grid()
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.xlabel("Time in months from start date")
plt.ylabel("Cumulative return")
plt.title("Subperiod:")
plt.show()


# Actual backtest test.
initial_fits = 3

test2, weights = bt.backtest_k(ind=data, mu_target=mu_target,m=initial_fits,l=1,K=1) # 36 trailing month window


time = pd.date_range(test2["Date"][0],test2["Date"][len(test2["Date"]) -1 ], freq = 'ME')
fig, ax = plt.subplots(layout='constrained')
ax.plot(time,test2["R_40/60"], label="R of 40/60")
ax.plot(time,test2["R_MV"], label="R of Mean-Var")
ax.plot(time,test2["R_MVL"], label="R of Mean-Var Leveraged")
ax.plot(time,test2["R_RP"], label="R of Risk Parity")
ax.plot(time,test2["R_RPL"], label="R of Risk Parity Leveraged")
ax.plot(time[initial_fits:], [(mu_target+1)**t for t in range(0,len(time)-initial_fits)],
         label = "Benchmark of 75Bps/Month")

ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Set a tick at the start of every year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.legend(framealpha=0.05,loc='center', bbox_to_anchor=(0.5, -0.5), ncol=3,prop={'size': 8})
plt.tight_layout()

plt.grid()
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.xlabel("Time in months from start date")
plt.ylabel("Cumulative return")
plt.title("Entire Period: 1990-2023")
plt.show()







# Actual backtest test.
initial_fits = 400

test2,e,r,t = bt.backtest_k(ind=data, mu_target=mu_target,m=initial_fits,l=1,K=1, olay = True) # 36 trailing month window


time = pd.date_range(test2["Date"][0],test2["Date"][len(test2["Date"]) -1 ], freq = 'ME')
fig, ax = plt.subplots(layout='constrained')
ax.plot(time,test2["R_40/60"], label="R of 40/60")
ax.plot(time,test2["R_MV"], label="R of Mean-Var")
ax.plot(time,test2["R_MVL"], label="R of Mean-Var Leveraged")
ax.plot(time,test2["R_RP"], label="R of Risk Parity")
ax.plot(time,test2["R_RPL"], label="R of Risk Parity Leveraged")
ax.plot(time[initial_fits:], [(mu_target+1)**t for t in range(0,len(time)-initial_fits)],
         label = "Benchmark of 75Bps/Month")

ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Set a tick at the start of every year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.legend(framealpha=0.05,loc='center', bbox_to_anchor=(0.5, -0.5), ncol=3,prop={'size': 8})
plt.tight_layout()

plt.grid()
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.xlabel("Time in months from start date")
plt.ylabel("Cumulative return")
plt.title("Entire Period: 1990-2023")
plt.show()



bond = pd.read_csv("Data_clean/bond_returns.csv")
stock = pd.read_csv("Data_clean/6_Portfolios_ME_Prior_12_2_returns.csv")
bond, stock =bond[["Date", "10YrReturns"]], stock[["Date", "Market Return"]]
stock["Market Return"] = stock["Market Return"] /100  
RF = pd.read_csv("Data_clean/FF_cleaned.csv")
data = pd.merge(bond,stock, how='left', on = "Date")

data = pd.merge(data.copy(),RF, 'left',on = "Date" )
data["RF"] = data["RF"] /100 # assumed this must hold
data = data.drop(columns = ["Mkt-RF", "SMB", "HML"])

# Test of markowitz implementation. 
test = data[(pd.Timestamp('1990-01-31') <= pd.to_datetime(data["Date"])) &
            (pd.Timestamp('1991-01-31') >= pd.to_datetime(data["Date"]))]

sigma = np.cov([test["RF"],test["10YrReturns"],test["Market Return"]])
mu = np.mean([test["RF"],test["10YrReturns"],test["Market Return"]],axis=1)
weights = Utils.get_weights2(mu,sigma,mu_target) 

rets = [weights[i] @ mu for i in range(0,len(weights))]
sigmas = [np.sqrt(weights[i] @ sigma @ weights[i])  for i in range(0,len(weights))]

mu_vec = np.array([mu_target + i*0.001 for i in range(-10,20)])
sigma_vec = [Utils.mv_analysis(mu, sigma,i)[1] for i in mu_vec]
mv_w = [Utils.mv_analysis(mu, sigma,i)[0] for i in mu_vec]

plt.plot(sigma_vec, mu_vec, color = "black",label = "Mean-Variance Frontier")
plt.plot(sigmas[0],rets[0],marker='o', color = "red", label = "40/60 Portfolio")
plt.plot(sigmas[1],rets[1],marker='o', color = "cyan", label = "Optimal Markowitz Portfolio")
plt.plot(sigmas[2],rets[2],marker='o', color = "green", label = "Optimal Markowitz with RF levered Portfolio")
plt.plot(sigmas[3],rets[3],marker='o', color = "yellow", label = "Risk Parity")
plt.plot(sigmas[4],rets[4],marker='o', color = "magenta", label = "Risk Parity leveraged")
plt.hlines(y = mu_target,xmin = 0, xmax = 0.07)
plt.xlim(left = 0.0,right= 0.07)

plt.legend()
plt.xlabel("Standard deviation")
plt.ylabel("Return")
plt.grid()
plt.show()
