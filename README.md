# Asset Allocation Exam Project

## Data cleaning
Data cleaning is conducted in the 'Data_cleaning'-type of files

## Reproducing findings:
This git contains our code (and data) to reproduce the findings presented in the project. 
1. To reproduce the part 1, run the notebook Task_1.ipynb 
2. To reproduce the part 2, run the notebook Task_2.ipynb
3. To reproduce the part 3, run the notebook Task_3.ipynb


## Underlying machinery
1. The Utils.py file contains several implementations called in the mentioned notebooks such as code for Mean-Variance optimization. 
3. Finally, the Backtest.py contains implementation of 2 backtest functions. One we denote naive, which takes applies a strategy found by fitting data on the entire period. The more reality near backtest finds the optimal strategies on the first m data points and trades according to this for 1 month. When rebalancing after 1 month, the first m+1 data points is used to invest for the next month and so on. 

