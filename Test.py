import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from scipy.optimize import minimize 
import Backtest as bt   
import matplotlib.dates as mdates

mu_target = 75 / 10000 # Target
bond = pd.read_csv("Data_clean/bond_returns.csv")
stock = pd.read_csv("Data_clean/6_Portfolios_ME_Prior_12_2_returns.csv")
bond, stock =bond[["Date", "10YrReturns"]], stock[["Date", "Market Return"]]
stock["Market Return"] = stock["Market Return"] /100  
RF = pd.read_csv("Data_clean/FF_cleaned.csv")
data = pd.merge(bond,stock, how='left', on = "Date")

data = pd.merge(data.copy(),RF, 'left',on = "Date" )
data["RF"] = data["RF"] /100 # assumed this must hold



# Actual backtest test.
initial_fits = 3

test2 = bt.backtest_k(ind=data, mu_target=mu_target,m=initial_fits,l=1,K=1) # 36 trailing month window


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



''''
test2,w0t,w1t,w2t = bt.backtest_naive(ind=data, mu_target=mu_target)

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

'''