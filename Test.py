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

# test of backtest. 
data1 = data.copy()
# Now perform the comparison
data1 = data1[pd.Timestamp('2010-01-31') <=pd.to_datetime(data1["Date"])]

test2,w0t,w1t,w2t = bt.backtest_naive(ind=data1, mu_target=mu_target)

time = pd.date_range(data1["Date"][0],data1["Date"][len(data["Date"]) -1 ], freq = 'ME')
fig, ax = plt.subplots()
ax.plot(time,data1["R_40/60"], label="R of 40/60")
ax.plot(time,data1["R_MV"], label="R of Mean-Var")
ax.plot(time,data1["R_MVL"], label="R of Mean-Var Leveraged")
ax.plot(time,data1["R_RP"], label="R of Risk Parity")
ax.plot(time,data1["R_RPL"], label="R of Risk Parity Leveraged")
ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Set a tick at the start of every year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend()
plt.grid()
plt.xticks(rotation=45, ha='right')

plt.xlabel("Time in months from start date")
plt.ylabel("Cumulative return")
plt.title("Subperiod:")
plt.show()
