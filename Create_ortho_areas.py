#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%
#Importing the relevant libraries

import numpy as np
import random
import csv
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from timeit import default_timer as timer
from tqdm import tqdm
import scipy.stats as st
import pylab 
from numpy import genfromtxt
plt.style.use('seaborn')


# In[2]:


#%%
#creating the mandelBort image firstly 
# adapted from http://numba.pydata.org/numba-doc/0.35.0/user/examples.html

from numba import int16, float32 , uint8 , uint32, complex64, prange,objmode
from numba.experimental import jitclass
from numba_progress import ProgressBar
spec = [
    ('minx', int16),               # a simple scalar field
    ('maxx', int16),
    ('miny', int16),
    ('maxy', int16),
    ('image', uint32[:,:]),          # an array field
    ('iterations', int16)
]


# In[3]:


@jitclass(spec)
class mandeL_plot:

    """ We are creating a class for the Mandelbrot set
        This will generate the mandelbrot set and convert the  values into 
        pixels, thereafter a plot will be made of the mandelbrot set
    """

    def __init__(self,minx, maxx, miny, maxy, image, iterations):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.image = image 
        self.iterations = iterations 
        
    def delete_workaround(self, arr, num):
        mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
        mask[np.where(arr == num)[0]] = False
        return arr[mask]
    
    def calc_convergent(self, x,y):
        iterations = self.iterations
        x = x
        y = y 

        i = 0
        c = complex(x,y)
        z = 0.0j
        for i in range(iterations):
            z = z*z + c 
            if (z.real *z.real + z.imag *z.imag >= 2):
                return i
        return 255
    
    def convert_to_pixels_plot(self):
        """this function converts the coordinates into pixels to generate 
        the images for plotting the mandelbrot set 

        Returns:
            image: returns the image of the mandelbrot set, this can then be used directlty 
            for plotting
        """
        image = self.image
        minx = self.minx
        maxx = self.maxx
        miny = self.miny
        maxy = self.maxy

        width = image.shape[0]
        height = image.shape[1]

        x_pixels = np.abs((maxx - minx))/width
        y_pixels = np.abs((maxy - miny))/height
        
        for i in range(width):
            real = minx + i*x_pixels
            for j in range(height):
                IM = miny + j*y_pixels
                color = self.calc_convergent(real, IM)
                image[i, j] = color

        return image

    def sampling_method_random(self, numsamples):
        """this function generates samples using numpy random 

        Args:
            numsamples (int): number of samples to be generated 

        Returns:
            complex list: list of generated samples in the imaginary and real space
        """
        
        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        samp = []

        for i in range(numsamples):
            sample = complex(random.uniform(real_min, real_max), random.uniform(imaginary_min, imaginary_max))
            samp.append(sample)
        
        return samp
        
    # @staticmethod    
    def within_mandel(self, iterations, c): 
        """this function takes the coordinates of the sample set and 
        and determines if the sample coordinate converges

        Args:
            iterations (int): number of iterations to be performed 
            c (complex coordinate): coordinate in mandelbrot space

        Returns:
            boolean: if the coordinate is convergent, return true
        """
        z = 0
        for i in range(iterations):
            if np.abs(z) > 2:
                return False
            else:
                z = z*z + c
        return True

    def compute_area_random(self, num_runs, nsamples, numiterations):

        """this function generates a list of the computed areas for a number of iterations,
        samples and number of runs 

        Returns:
            list(float32): generates and returns the list of the computed areas for all runs 
        """

        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        areas = []

        for i in range(num_runs):
            in_mandel = 0
            total_drawn = nsamples
            area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)

            samples = self.sampling_method_random(nsamples)

            for c in samples:
                if (self.within_mandel(numiterations, c)):
                    in_mandel += 1

            ratio_inmandel = (in_mandel/total_drawn)
            area_mandel = ratio_inmandel*area_T        

            areas.append(area_mandel)

        return areas


    def return_area_matrix_constant_iterations_random(self, num_runs, num_samples, num_iterations, areas_matrix):
        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of samples

        Args:
            num_runs (int): number of runs/simulations which need to be performed
            num_samples (int): number of samples drawn to evaluate the area of the mandelbrot set
            num_iterations (int): number of iterations to be satisfied to compute convergence
            areas_matrix (numpy array(2D)): area matrix for storing simulation results 

        Returns:
            numpy array(2D): returns the area matrix for all simulations 
        """
        am = areas_matrix
        for i in num_samples:
            area = self.compute_area_random(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am

    def return_area_matrix_constant_samples_random(self, num_runs, num_samples, num_iterations, areas_matrix):
        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of iterations

        Args:
            num_runs (int): number of runs/simulations which need to be performed
            num_samples (int): number of samples drawn to evaluate the area of the mandelbrot set
            num_iterations (int): number of iterations to be satisfied to compute convergence
            areas_matrix (numpy array(2D)): area matrix for storing simulation results 

        Returns:
            numpy array(2D): returns the area matrix for all simulations 
        """
        
        am = areas_matrix
        for i in num_iterations:
            area = self.compute_area_random(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am


    def calculate_required_samples(self, num_runs, num_iterations, d):
        """this function determines the number of samples 

        Args:
            num_runs (_type_): _description_
            num_iterations (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """

        samples = 900
        k = 100
        std = 0
        areas = 0
        while (d < k):
            area = self.compute_area_random(num_runs, samples, num_iterations)
            area = np.array(area, dtype = np.float32)
            areas = np.mean(area)
            std_dev = np.std(area)
            std = std_dev
            k = std/np.sqrt(samples)
            samples += 100

        return samples, std, areas

    def calculate_required_number_of_runs(self, num_iterations, num_samples):

        num_runs = 100
        k = 100
        while (d < std):
            area  = self.compute_area_random(num_runs, num_samples, num_iterations)
            area = np.array(area, dtype = np.float32)
            std_dev = np.std(area)
            std = std_dev
            num_runs += 500

        return samples, std

    def generate_LHS(self, num_samples):
        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        # make x an y spaces to form the hypercube 
        x = np.linspace(real_min, real_max, num_samples+1)
        y = np.linspace((imaginary_min), (imaginary_max), num_samples+1)

        lhs_points = np.empty(shape=(num_samples, 2))
        for i in range(num_samples):
            lhs_points[i, 0] = np.random.uniform(x[i], x[i + 1])
            lhs_points[i, 1] = np.random.uniform(y[i], y[i + 1])

        np.random.shuffle(lhs_points[:, 1])
        samples = [complex(lhs_points[n, 0], lhs_points[n , 1]) for n in range(len(lhs_points))]

        return samples


    def compute_area_LHS(self, num_runs, nsamples, numiterations):

        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        areas = []

        total_drawn = nsamples
        area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)

        for i in range(num_runs):
            in_mandel = 0
            samples = self.generate_LHS(nsamples)
            for c in samples:
                if (self.within_mandel(numiterations, c)):
                    in_mandel += 1

            ratio_inmandel = (in_mandel/total_drawn)
            area_mandel = ratio_inmandel*area_T        

            areas.append(area_mandel)

        return areas
    
    
    def generate_ortho(self, num_samples):
    
        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        # make x an y spaces to form the 2d hypercube just like LHS
        x = np.linspace(real_min, real_max, num_samples+1)
        y = np.linspace((imaginary_min), (imaginary_max), num_samples+1)

        x_ortho_ranges = np.empty(shape=(len(x), 2))
        y_ortho_ranges = np.empty(shape=(len(y), 2))

        for i in range(1, len(x)):
            x_ortho_ranges[i] = (x[i-1], x[i])
            y_ortho_ranges[i] = (y[i-1], y[i])
        
        x_ortho_ranges = x_ortho_ranges[1:,:]
        y_ortho_ranges = y_ortho_ranges[1:,:]
        
        
        A = np.zeros(num_samples ** 2).reshape(num_samples, num_samples)
        # make the subspaces in the hypercube, this is specific for orthogonal
        x_subspaces = np.arange(0, num_samples+np.sqrt(num_samples), np.sqrt(num_samples))
        y_subspaces = np.arange(0, num_samples+np.sqrt(num_samples), np.sqrt(num_samples))

        block_interval = np.empty(shape = (len(x_subspaces-1), 2))

        for i in range(1, len(x_subspaces)):
            block_interval[i] = (int(x_subspaces[i-1]), int(x_subspaces[i]))
        
        block_interval = block_interval[1:,:]
        
        # Make the indices that will be shuffled each run
        x_indices = np.arange(0, num_samples, 1)
        y_indices = np.arange(0, num_samples, 1)
        
        np.random.shuffle(x_indices)
        np.random.shuffle(y_indices)

        samples = []

        for i in block_interval:
            for j in block_interval:
                # create list with individual blocks in the interval range
                x_blocks = np.arange(j[0], j[1], 1)
                y_blocks = np.arange(i[0], i[1], 1)
                
                # loop through available indices
                for k in x_indices:
                # if an available index is in slice of indices
                
                    if k in x_blocks:
                        # set x coordinate to available index
                        coordinate_x = k
                        # remove available index
                        x_indices = self.delete_workaround(x_indices, k)
                        break

                for k in y_indices:
                    if k in y_blocks:
                        coordinate_y = k
                        y_indices = self.delete_workaround(y_indices, k)
                        break
                
                A[coordinate_y][coordinate_x] = 1
                
                x_sample_range = x_ortho_ranges[coordinate_x]
                y_sample_range = y_ortho_ranges[-(coordinate_y+1)]

                sample = complex(np.random.uniform(x_sample_range[0], x_sample_range[1]), 
                                 np.random.uniform(y_sample_range[0], y_sample_range[1]))
                samples.append(sample)
                
                #print(A[(i[0]):(i[1]), (j[0]):(j[1])])
        #print(A)
        return samples
    
    def compute_area_ortho(self, num_runs, nsamples, numiterations):

        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        areas = []

        for i in range(num_runs):
            in_mandel = 0
            total_drawn = nsamples
            area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)

            samples = self.generate_ortho(nsamples)

            for c in samples:
                if self.within_mandel(numiterations, c):
                    in_mandel += 1

            ratio_inmandel = (in_mandel/total_drawn)
            area_mandel = ratio_inmandel*area_T        

            areas.append(area_mandel)

        return areas

        
    def return_area_matrix_constant_iterations_LHS(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_samples:
            area = self.compute_area_LHS(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am

    def return_area_matrix_constant_samples_LHS(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_iterations:
            area = self.compute_area_LHS(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am
    
    def return_area_matrix_constant_iterations_ortho(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_samples:
            area = self.compute_area_ortho(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am
    
    def return_area_matrix_constant_samples_ortho(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_iterations:
            area = self.compute_area_ortho(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am


# In[4]:


RE_START = -2.0
RE_END = 1.0
IM_START = -1.2
IM_END = 1.2
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
its = 1000

mandel= mandeL_plot(RE_START, RE_END, IM_START, IM_END, image, its)


# In[ ]:


# %%
""" We are now going to test how the average of the mandelbrot set converges 
as we increase the number of iterations while keeping the number of samples 
constant
"""
#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# conducting the second test #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
num_runs = 1000
num_samples = 2000
num_iterations = np.arange(100, 12000, 100, dtype = np.int16)
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%
#%#%#%#%#%#%# this has already been performed #%#%#%#%#%#%#%#%#%%#%#%#%#%#%#
areas_ortho_its =mandel.return_area_matrix_constant_samples_ortho(num_runs, num_samples, num_iterations, areas_matrix)
np.savetxt("AM_Ortho_ITS.csv", areas_ortho_its, delimiter=",")


# In[ ]:


num_runs = 1000
# this will be used to generate the larger matrix
num_samples = np.arange(10, 12000, 100, dtype = np.int16)
num_iterations = 2500
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)

areas_ortho =mandel.return_area_matrix_constant_iterations_ortho(num_runs, num_samples, num_iterations, areas_matrix)
np.savetxt("AM_Ortho.csv", areas_ortho, delimiter=",")
#%%


# In[ ]:




