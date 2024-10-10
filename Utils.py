import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    

    

