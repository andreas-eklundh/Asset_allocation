### Test - basic packages:

import numpy as np


A = np.array([[1,0,0],
              [0,2,0],
              [0,0,1]])

B = np.array([3,4,5])

print(A @ B)


import pandas as pd

df = pd.DataFrame(A, columns=["A", "B", "C"])

df = df.copy() + 2


print(df)

print("Hello World!")