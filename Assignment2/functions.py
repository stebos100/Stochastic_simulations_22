#%%
#importing the various libraries for use

import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy import integrate
import scipy.stats as st

#%%
# functions outside of the class under analysis 

def calculate_confidence_interval(matrix):
    """This function generates a 95% confidence interval for a matrix of areas calculated using MC simulations

    Args:
        matrix (numpy array 2D): matrix containing all area computations

    Returns:
        numpy array: array of confidence intervals for the average of each simulation
    """

    cis = np.ones(shape = (1,2))

    for i in matrix:
        data = i 
        interval = np.array(st.t.interval(alpha=0.95, df=(matrix.shape[1])-1, loc=np.mean(data), scale=st.sem(data)))
        interval = interval.reshape(1,2)
        cis = np.vstack((cis, interval))

    return cis

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
    '''This function calculates the theoretical prediction for M/D/n'''
    EXPECTATION_mmn = theoretical_mmn(rho, mu, n)
    C = 0
    return (C + 1) / 2 * EXPECTATION_mmn

def longtail_pred(rho, mu, n):
    '''This function calculates the theoretical prediction for part hyper exponential M/H/n'''

    p = 0.25
    ux = 5
    uy = 0.75 * mu * ux / (ux - 0.25 * mu)
    C = (p * 1/ux ** 2 + (1 - p) * p * (1/ux - 1/uy) ** 2 + (1 - p) * 1/uy ** 2) / (p * 1/ux + (1 - p) * 1/uy) ** 2
    EXPECTATION_mmn = theoretical_mmn(rho, mu, n)
    return (C + 1) / 2 * EXPECTATION_mmn

def SPTF(rho, mu, n, upp=100):
    '''A function to calculate the theoretical prediction for M/M/n-SPTF'''
    if n > 1: 
        return None
    def integrand(x):
        return rho * np.exp(-x * mu) / (1 - (1 - np.exp(-x * mu) * (1 + x * mu)) * rho) ** 2
    I = integrate.quad(integrand, 0, upp)
    return I[0]

def return_stds_formatting(std_1000, std_5000, std_10000, std_20000, std_50000, std_75000, std_100000, p_range):
    """this function takes the standard deviations for the various simulations and reformats them 
    to comply with what is desired ie the standard deviations at a specific utlization rate 

    Args:
        std_1000 (list): a list containin the std deviations for 1000 samples and 50 runs
        std_5000 (list): a list containin the std deviations for 5000 samples and 50 runs
        std_10000 (list): a list containin the std deviations for 10000 samples and 50 runs
        std_20000 (list): a list containin the std deviations for 20000 samples and 50 runs
        std_50000 (list): a list containin the std deviations for 50000 samples and 50 runs
        std_75000 (list): a list containin the std deviations for 75000 samples and 50 runs
        std_100000 (list): a list containin the std deviations for 100000 samples and 50 runs
        p_range (list): a list containin the relavant p values for analysis

    Returns:
         lists(n = 1,2,4 servers): returns a list for all the various server values 
    """
    stds_results= np.zeros((1, 7))
    stds_results_2= np.zeros((1, 7))
    stds_results_3= np.zeros((1, 7))
    p_plot_range = []
    for i in range(4,10):
        
        res = []
        res1 = []
        res2 = []
        p_plot_range.append(p_range[i])
        res.append([std_1000[i],std_5000[i], std_10000[i], std_20000[i], std_50000[i], \
            std_75000[i], std_100000[i]])

        res1.append([std_1000[i+10],std_5000[i+10], std_10000[i+10], std_20000[i+10], std_50000[i+10], \
        std_75000[i+10], std_100000[i+10]])

        res2.append([std_1000[i+20],std_5000[i+20], std_10000[i+20], std_20000[i+20], std_50000[i+20], \
        std_75000[i+20], std_100000[i+20]])
        
        res = np.array(res)
        res = res.reshape(1,7)
        res1 = np.array(res1)
        res1 = res1.reshape(1,7)
        res2 = np.array(res2)
        res2 = res2.reshape(1,7)
        stds_results = np.concatenate((stds_results, res))
        stds_results_2 = np.concatenate((stds_results_2, res1))
        stds_results_3 = np.concatenate((stds_results_3, res2))

    stds_results = stds_results[1:]
    stds_results_2 = stds_results_2[1:]
    stds_results_3 = stds_results_3[1:]

    return stds_results, stds_results_2, stds_results_3, p_plot_range

def return_avgs_formatting(avg_1000, avg_5000, avg_10000, avg_20000, avg_50000, avg_75000, avg_100000, p_range):
    """this function takes the average waiting times for the various simulations and reformats them 
    to comply with what is desired ie the standard deviations at a specific utlization rate 

    Args:
        avg_1000 (list): a list containin the avg waiting times for 1000 samples and 50 runs
        avg_5000 (list): a list containin the avg waiting times for 5000 samples and 50 runs
        avg_10000 (list): a list containin the avg waiting times for 10000 samples and 50 runs
        avg_20000 (list): a list containin the avg waiting times for 20000 samples and 50 runs
        avg_50000 (list): a list containin the avg waiting times for 50000 samples and 50 runs
        avg_75000 (list): a list containin the avg waiting times for 75000 samples and 50 runs
        avg_100000 (list): a list containin the avg waiting times for 100000 samples and 50 runs
        p_range (list): a list containin the relavant p values for analysis

    Returns:
            lists(n = 1,2,4 servers): returns a list for all the various server values 
    """

    avgs_results= np.zeros((1, 7))
    avgs_results_2= np.zeros((1, 7))
    avgs_results_3= np.zeros((1, 7))
    p_plot_range = []
    for i in range(4,10):
        
        res = []
        res1 = []
        res2 = []
        p_plot_range.append(p_range[i])
        res.append([avg_1000[i],avg_5000[i], avg_10000[i], avg_20000[i], avg_50000[i], \
            avg_75000[i], avg_100000[i]])

        res1.append([avg_1000[i+10],avg_5000[i+10], avg_10000[i+10], avg_20000[i+10], avg_50000[i+10], \
        avg_75000[i+10], avg_100000[i+10]])

        res2.append([avg_1000[i+20],avg_5000[i+20], avg_10000[i+20], avg_20000[i+20], avg_50000[i+20], \
        avg_75000[i+20], avg_100000[i+20]])
        
        res = np.array(res)
        res = res.reshape(1,7)
        res1 = np.array(res1)
        res1 = res1.reshape(1,7)
        res2 = np.array(res2)
        res2 = res2.reshape(1,7)
        avgs_results = np.concatenate((avgs_results, res))
        avgs_results_2 = np.concatenate((avgs_results_2, res1))
        avgs_results_3 = np.concatenate((avgs_results_3, res2))

    avgs_results = avgs_results[1:]
    avgs_results_2 = avgs_results_2[1:]
    avgs_results_3 = avgs_results_3[1:]

    return avgs_results, avgs_results_2, avgs_results_3, p_plot_range