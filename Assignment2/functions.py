#%%
#importing the various libraries for use

import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy import integrate

#%%
# functions outside of the class under analysis 

def theoretical_mmn(rho, mu, n):
    def W(n, rho):
        def B(n, rho):
            B = 1
            for i in range(1, n+1):
                B = rho * B / (i + rho * B)
            return B
        B2 = B(n-1, n* rho)
        return rho * B2 / (1 - rho + rho * B2)
    w = W(n, rho)
    return w / (n * mu) * ( 1 / (1 - rho))

def MDn(rho, mu, n):
    '''A function to calculate the theoretical prediction for M/D/n'''
    EW_MMn = theoretical_mmn(rho, mu, n)
    C = 0
    return (C + 1) / 2 * EW_MMn

def longtail_pred(rho, mu, n):
    '''A function to calculate the theoretical prediction for M/l/n'''
    
    p = 0.25
    ux = 5
    uy = 0.75 * mu * ux / (ux - 0.25 * mu)
    C = (p * 1/ux ** 2 + (1 - p) * p * (1/ux - 1/uy) ** 2 + (1 - p) * 1/uy ** 2) / (p * 1/ux + (1 - p) * 1/uy) ** 2
    EW_MMn = theoretical_mmn(rho, mu, n)
    return (C + 1) / 2 * EW_MMn

def SPTF(rho, mu, n, upp=100):
    '''A function to calculate the theoretical prediction for M/M/n-SPTF'''
    if n > 1: 
        return None
    def integrand(x):
        return rho * np.exp(-x * mu) / (1 - (1 - np.exp(-x * mu) * (1 + x * mu)) * rho) ** 2
    I = integrate.quad(integrand, 0, upp)
    return I[0]
