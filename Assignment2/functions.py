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

def task(env, server, processing_time, waiting_time):
    """task arrives, is served and leaves."""
    arrive = env.now
    with server.request() as req:
        yield req
        waiting_time.append(env.now-arrive)
        yield env.timeout(processing_time)
        
def short_task(env, server, processing_time, waiting_time):
    """task arrives, is served with a priority equal to the processing time and leaves."""
    arrive = env.now
    with server.request(priority=processing_time) as req:
        yield req
        waiting_time.append(env.now-arrive)
        yield env.timeout(processing_time)
        
def deterministic(x):
    '''Returns the deterministic time corresponding to a capacity of x'''
    return 1/x

def longtail(x):
    '''Returns a long tail distribution with mean 1/x 
    where 25% has an exponential distribution with mean processing capacity = 3
    and 75% an exponential with a mean so that the mean of the distribution is 1/x '''
    mu_big = 3
    mu_small = 0.75*x*mu_big/(mu_big-0.25*x)
    a = random.random()
    if a < 0.25:
        n = random.expovariate(mu_big)
    else:
        n = random.expovariate(mu_small)
    return n

def MDn(rho, mu, n):
    '''A function to calculate the theoretical prediction for M/D/n'''
    EW_MMn = theoretical_mmn(rho, mu, n)
    C = 0
    return (C + 1) / 2 * EW_MMn

def longtail_pred(rho, mu, n):
    '''A function to calculate the theoretical prediction for M/l/n'''
    # print(mu, rho)
    p = 0.25
    ux = 3
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
