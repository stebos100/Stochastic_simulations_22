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


# In[14]:


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


#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% FUNCTIONS outside of the class #%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%%#%#%#%%#
def calculate_confidence_interval(matrix):

    cis = np.ones(shape = (1,2))

    for i in matrix:
        data = i 
        interval = np.array(st.t.interval(alpha=0.95, df=(num_runs)-1, loc=np.mean(data), scale=st.sem(data)))
        interval = interval.reshape(1,2)
        cis = np.vstack((cis, interval))

    return cis


# In[5]:


#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% Creating the mandel brot set plot #%#%#%#%#%#%#%#%#%#%#%#%#%#
'''Creating the mandel brot set plot'''
minx =  -2.0
maxx = 1.10
miny = -1.98 
maxy = 1.98
image = np.zeros((8000 * 2,  6000* 2), dtype = np.uint32)
iterations = 200

mandel= mandeL_plot(minx, maxx, miny, maxy, image, iterations)
s = timer()
im = mandel.convert_to_pixels_plot()
e = timer()
print(e-s)


# In[6]:


#%%
#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%# plotting the figure and saving it #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(im.T,cmap='gist_earth')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Im', fontsize = 14)
ax.set_xlabel("Re", fontsize = 14)
plt.show()
plt.savefig("mandel_plot.png", dpi = 700,  bbox_inches = 'tight', format = 'png' )
#%%


# In[7]:


#%#%#%#%#%#%##%#%#%##%#%#%# creating a second plot which zooms in on the intersting areas #%#%#%#%#%#%#%#%#
minx =  -1.15
maxx = -0.7
miny = -1.0 
maxy = -0.4
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
iterations = 100

mandel= mandeL_plot(minx, maxx, miny, maxy, image, iterations)
s = timer()
im = mandel.convert_to_pixels_plot()
e = timer()
print(e-s)


# In[8]:


#%%
#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%# plotting the figure and saving it #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

get_ipython().run_line_magic('matplotlib', 'widget')

fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(im,cmap='gist_earth')
ax.set_xlim(6200,8000)
ax.set_ylim(12000,14500)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Im', fontsize = 14)
ax.set_xlabel("Re", fontsize = 14)
plt.show()
plt.savefig("mandel_plot_second.pdf", dpi = 900,  bbox_inches = 'tight', format = 'pdf' )


# In[9]:


#%#%#%# start here #%#%#%#%#%#%#%#
#%#%%#%#%#%#%#%#%#%#%#%#%#%#% Creating a second class instance for testing convergence #%#%#%#%#%#%#%#%#%#%#%#
RE_START = -2.0
RE_END = 1.0
IM_START = -1.2
IM_END = 1.2
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
its = 1000

mandel= mandeL_plot(RE_START, RE_END, IM_START, IM_END, image, its)


# In[10]:


#%%
#%#%#%#%#%#%#%#%#%#%#%#%# CHECKING THE NUMBER OF SAMPLES REQUIRED #%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#%#%#%# THIS HAS ALREADY BEEN PERFORMED #%#%#%#%#%#%#%#%#%#%%#%##

# iterations = np.arange(1000, 10000, 100)
# samples_varying_iterations = np.array([])
# stds = np.array([])
# areas = np.array([])

# for i in tqdm(iterations):
#     samples_varying_iterations = np.append(samples_varying_iterations, mandel.calculate_required_samples(1000, i, 0.005)[0])
#     stds = np.append(stds, mandel.calculate_required_samples(1000, i, 0.005)[1])
#     areas = np.append(areas,  mandel.calculate_required_samples(1000, i, 0.005)[2])

# #%%
# data_for_iterations = np.zeros(shape= (1, 90))
# data_for_iterations = np.concatenate((data_for_iterations, samples_varying_iterations.reshape(1,90)))
# data_for_iterations = np.concatenate((data_for_iterations, stds.reshape(1,90)))
# data_for_iterations = np.concatenate((data_for_iterations, areas.reshape(1,90)))
# data_for_iterations = np.concatenate((data_for_iterations, iterations.reshape(1,90)))
# data_for_iterations = data_for_iterations[1:]
# np.savetxt("num_samples.csv", data_for_iterations, delimiter=",")


# In[ ]:


#%%
area = mandel.compute_area_random(1000, 10000, 10000)
#%%
plt.style.use('seaborn')
st.probplot(area, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%# secondary check - histogram #%#%#%#%#%#%%#%#%#%#%#%#%%##
from scipy.stats import norm
mu, std = norm.fit(area) 
xmin, xmax = np.min(area), np.max(area)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.hist(my_data[-1], bins = 'auto', density=True)
plt.xlabel('Area of mandelBrot set',  fontsize = 14)
plt.ylabel('Number of occurences', fontsize = 14)
plt.title('histogram of data generated', fontsize  = 16)


# In[ ]:


#%%
from numpy import genfromtxt
samples_req = genfromtxt('num_samples.csv', delimiter=',')

#%%
"""using the above plots will determine the number of samples required for the 
    simulations where we test the convergence of the area as we increase iterations
    and as we  increase the number of samples. 

    from the above experiment we see that 900 samlples is sufficient and will therefore
    use 1000 for future experiments 
"""


# In[ ]:


#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[0,:], label ='number of samples required, 1000 runs for each sample set')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Number of samples', fontsize = 14)
plt.title('Number of samples required for desired Standard deviation', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


# %%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[1,:], linewidth = 3)
plt.fill_between(samples_req[3,:], np.max(samples_req[1,:]), np.min(samples_req[1,:]),  color = 'blue', alpha = 0.1, label = '{:.4f} minimum, {:.4f} maximum'.format(np.min(samples_req[1,:]), np.max(samples_req[1,:])))
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('sample standard deviation', fontsize = 14)
plt.title('Standard deviation for increasing iterations', fontsize = 16)
plt.legend(fontsize = 12)
plt.ylim(0.08, 0.105)


# In[ ]:


#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[2,:])
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Sample area of MandelBrot set', fontsize = 14)
plt.title('Sample area for increasing iterations', fontsize = 16)
plt.legend(fontsize = 12)
ax.axhline(y = 1.5064, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
ax.axhline(y = np.mean(samples_req[2,:]), color = 'k', linestyle = '-.', label = 'Average area of the MandelBrot set')
plt.ylim(1.49, 1.52)
plt.legend(fontsize = 12)


# In[ ]:


#%%
"""
now that we have established the baseline for the minimum number of samples
we are going to look at the normality of the plot for 10 000 samples and 10 000 iterations
"""
#%%
area = mandel.compute_area_random(1000, 10000, 10000)


# In[ ]:


#%%
plt.style.use('seaborn')
st.probplot(area, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%# secondary check - histogram #%#%#%#%#%#%%#%#%#%#%#%#%%##
from scipy.stats import norm
mu, std = norm.fit(area) 
xmin, xmax = np.min(area), np.max(area)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.hist(my_data[-1], bins = 'auto', density=True)
plt.xlabel('Area of mandelBrot set',  fontsize = 14)
plt.ylabel('Number of occurences', fontsize = 14)
plt.title('histogram of data generated', fontsize  = 16)


# In[ ]:


#%%
"""
we will now investigate how the number of samples influence the convergence behaviour 
of the area calculations
"""

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#% using the create area matrix function to store the data #%#%#%#%#%#%#%#%#%
num_runs = 1000
num_samples = np.arange(10, 12000, 100, dtype = np.int16)
num_iterations = 2000
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% storing the data #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%%#
#%#%#%#%#%#%#%#%#%#%#%#%#%# I have edited this out, its already been done #%#%#%#%#%#%%#%#%#%#%
# import pandas as pd 
# areas1 = mandel.return_area_matrix(num_runs, num_samples, num_iterations, areas_matrix, 1)
# pd.DataFrame(areas).to_csv("area_matrix_one.csv")
# np.savetxt("AM_one.csv", areas, delimiter=",")


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# example of how to import the data #%#%#%#%#%#%#%#%#%#%#%
from numpy import genfromtxt
my_data = genfromtxt('AM_one.csv', delimiter=',')
my_data.shape


# In[ ]:


#%%
"""firstly we want to evaluate the normality of the results to see if this is satisfactory 
from here we can then include confidence intervals and use normal distrbution 
testing methods 
"""
#%#%#%#%#%#%#%#%#%#%# checking the normality of the plots using a QQ and histogram #%#%#%#%#%#%#
plt.style.use('seaborn')
st.probplot(my_data[-1], dist="norm", plot=pylab)
pylab.show()


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%# secondary check - histogram #%#%#%#%#%#%%#%#%#%#%#%#%%##
from scipy.stats import norm
mu, std = norm.fit(my_data[-1]) 
xmin, xmax = np.min(my_data[-1]), np.max(my_data[-1])
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.hist(my_data[-1], bins = 'auto', density=True)
plt.xlabel('Area of mandelBrot set',  fontsize = 14)
plt.ylabel('Number of occurences', fontsize = 14)
plt.title('histogram of data generated', fontsize  = 16)


# In[ ]:


#%%
"""Going to now test the convergent behaviour as we increased the number of 
samples while the number of iterations remain constant  
"""

#%#%#%#%#%#%#%#%#%#%#%#%#%#% generating the mean of all simulations along the rows #%#%#%#%#%#%
mean = np.mean(my_data, axis = 1)
mean.shape
num_samples = np.arange(10, 12000, 100)
cis = calculate_confidence_interval(my_data[1:])
cis = cis[1:]


# In[ ]:


#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% plotting the mean to the true solution to test the convergence of the solution #%#%#%#%#%#%#%#%#%#%#%#%#%#%#
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean[1:]), label = 'Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5064, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_samples, cis[:, :1].reshape(120), cis[:, 1:].reshape(120),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# plotting the standard devuation  of the resulst obtained #%#%#%#%#%#%#%#
std = np.std(my_data, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (std[1:]), label = 'Standard deviation')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)


# In[ ]:


# %%
#%#%#%#%#%#%#%#%#%#%#%# plotting the computational cost associated with random sampling #%#%#%#%#%#
num_iterations = np.arange(100, 10000, 100, dtype = np.int16)
time = np.array([])
for i in tqdm(num_iterations):
    s = timer()
    area = mandel.compute_area_random(100, 1000, i)
    e = timer()
    time = np.append(time, (e-s))


# In[ ]:


#%%
fig, axs  = plt.subplots(figsize = (8,8))
axs.plot(num_iterations, time)
plt.xlabel('number of iterations', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with random sampling', fontsize = 16)


# In[ ]:


#%%
""" We are now going to test how the average of the mandelbrot set converges 
as we increase the number of iterations while keeping the number of samples 
constant
"""
#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# conducting the second test #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
num_runs = 1000
num_samples = 2000
num_iterations = np.arange(10, 12000, 100, dtype= np.int16)
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%


# In[ ]:


#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# convergence for increasing iterations #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
# areas2 = mandel.return_area_matrix_constant_samples_random(num_runs, num_samples, num_iterations, areas_matrix)
# areas2 = areas2[1:]
# np.savetxt("AM_TWO_RETEST.csv", areas2, delimiter=",")
#%%
my_data2 = genfromtxt('AM_TWO_RETEST.csv', delimiter=',')
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%# IMPORTING THE DATA #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#
# my_data2 = genfromtxt('AM_two.csv', delimiter=',')
#%#%#%#%#%#%#%#%#%#%#%#%#%#% generating the mean of all simulations along the rows #%#%#%#%#%#%
#%%
my_data2 = my_data2[1:]
#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% plotting the mean to the true solution to test the convergence of the solution #%#%#%#%#%#%#%#%#%#%#%#%#%#%#
mean2 = np.mean(my_data2, axis = 1)


# In[ ]:


#%%
cis2 = calculate_confidence_interval(my_data2)
cis2 = cis2[1:]


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations[1:], (mean2), label = 'Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5064, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations[1:], cis2[:, :1].reshape(119), cis2[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


#%%
std_2 = np.std(my_data2, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations[1:], (std_2), label = 'Standard deviation')
plt.ylim(0.05, 0.066)
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)


# In[15]:


#%%
#%#%#%#%#%#%#%#%#%#%#%# 
#%%
#%#%#%#%#%#%#%#%#%# firstly  show the difference between LHS, Ortho and random #%#%#%#%#%#%#%#%#%%#
"""
generating 400 samples of pure random sampling vs lhs 
going to analyse the histograms of the plots and show how the method differs
with a more balanced dispersion of points 

"""
RE_START = -2.0
RE_END = 1.0
IM_START = -1.2
IM_END = 1.2
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
its = 100

mandel= mandeL_plot(RE_START, RE_END, IM_START, IM_END, image, its)


# In[6]:


ortho_samples = mandel.generate_ortho(500)
samples = mandel.sampling_method_random(500)
lhs_samples = mandel.generate_LHS(500)
#%%
x = [ele.real for ele in ortho_samples]
x1 = [ele.real for ele in lhs_samples]
x2 = [ele.real for ele in samples]
y = [ele.imag for ele in ortho_samples]
y1 = [ele.imag for ele in lhs_samples]
y2 = [ele.imag for ele in samples]


kwargs = dict(histtype='stepfilled', alpha=0.3)
fig, ax = plt.subplots(figsize = (8,8))
ax.hist(x, **kwargs, label = 'real')
ax.hist(y, **kwargs, label = 'imaginary')
ax.set_xlabel('values', fontsize = 14)
ax.set_ylabel('count', fontsize = 14)
plt.title('histogram of Orthogonal sampling', fontsize = 16)
plt.legend(fontsize = 12)

fig, ax1 = plt.subplots(figsize = (8,8))
ax1.hist(x1, **kwargs, label = 'real')
ax1.hist(y1, **kwargs, label = 'imaginary')
ax1.set_xlabel('values', fontsize = 14)
ax1.set_ylabel('count', fontsize = 14)
plt.title('histogram of Latin Hypercube sampling', fontsize = 16)
plt.legend(fontsize = 12)

fig, ax2 = plt.subplots(figsize = (8,8))
ax2.hist(x2, **kwargs, label = 'real')
ax2.hist(y2, **kwargs, label = 'imaginary')
ax2.set_xlabel('values', fontsize = 14)
ax2.set_ylabel('count', fontsize = 14)
plt.title('histogram of Random sampling', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


# -------------------------------- ORTHO -----------------------------------
#%#%#%#%#%#%#%#%#%# TESTING THE AREA #%#%#%#%#%#%#%#%#
num_runs = 1000
# this will be used to generate the larger matrix
num_samples = np.arange(10, 12000, 100, dtype = np.int16)
num_iterations = 2500
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%
areas_ortho = mandel.return_area_matrix_constant_iterations_ortho(num_runs, num_samples, num_iterations, areas_matrix)
np.savetxt("AM_ortho.csv", areas_ortho, delimiter=",")


# In[ ]:


#%%
areas_ortho = genfromtxt('AM_ortho.csv', delimiter=',')

#%%
# just neglecting the initialization of the matrix, ie the first row consists of zeros 
areas_ortho = areas_ortho[1:,]
areas_ortho.shape


# In[ ]:


#%%
#generating the confidence interval for the plots, neglecting the first one as its just one
mean_ortho = np.mean(areas_ortho, axis = 1)
cis3 = calculate_confidence_interval(areas_ortho)
cis3 = cis3[1:]


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_ortho), label = 'Average Area Ortho method for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_samples, cis3[:, :1].reshape(cis3.shape[0]), cis2[:, 1:].reshape(cis3.shape[0]),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


#%#%#%#%#%#%#%#%%# convergent behaviour of the standard deviation #%#%#%#%#%%#%#%#%#%%#%#%%#
std_ortho = np.std(areas_ortho, axis =1)


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_ortho, label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)


# In[ ]:


#---------------------------------------LHS-------------------------------------------------------

#%%
"""going to now generate the area matrix for LHS sampling and then plot the same figures as before
testing the convergent behaviour as we increase the number of samples for each run 
"""
#%#%#%#%#%#%#%#%#%# TESTING THE AREA #%#%#%#%#%#%#%#%#
num_runs = 1000
# this will be used to generate the larger matrix
num_samples = np.arange(10, 12000, 100, dtype = np.int16)
num_iterations = 2500
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%
# areas_lhs =mandel.return_area_matrix_constant_iterations_LHS(num_runs, num_samples, num_iterations, areas_matrix)
#np.savetxt("AM_LHS.csv", areas_lhs, delimiter=",")
#%%
areas_lhs = genfromtxt('AM_LHS.csv', delimiter=',')

#%%
# just neglecting the initialization of the matrix, ie the first row consists of zeros 
areas_lhs = areas_lhs[1:,]
areas_lhs.shape


# In[ ]:


#%%
#generating the confidence interval for the plots, neglecting the first one as its just one
mean_lhs = np.mean(areas_lhs, axis = 1)
cis2 = calculate_confidence_interval(areas_lhs)
cis2 = cis2[1:]


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_lhs), label = 'Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_samples, cis2[:, :1].reshape(cis2.shape[0]), cis2[:, 1:].reshape(cis2.shape[0]),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:



#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_lhs), label = 'LHS Average Area for for 1000 runs, 2000 iterations')
ax.plot(num_samples, (mean[1:]), label = 'Pure random Sampling Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_samples, cis2[:, :1].reshape(cis2.shape[0]), cis2[:, 1:].reshape(cis2.shape[0]),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


#%#%#%#%#%#%#%#%%# convergent behaviour of the standard deviation #%#%#%#%#%%#%#%#%#%%#%#%%#
std_LHS = np.std(areas_lhs, axis =1)


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)


# In[ ]:


#%%
#%#%#%#%#%#%# comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
ax.plot(num_samples, std[1:], label = 'Standard deviation Random sampling')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)


# In[ ]:


#%%
#%#%#%#%#%#%# comparing to pure random sampling and ortho #%#%#%#%#%#%#%#%#%#%#
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_ortho, label ="Standard deviation ortho")
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
ax.plot(num_samples, std[1:], label = 'Standard deviation Random sampling')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)


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
# areas_lhs_its =mandel.return_area_matrix_constant_samples_LHS(num_runs, num_samples, num_iterations, areas_matrix)
# np.savetxt("AM_LHS_ITS.csv", areas_lhs_its, delimiter=",")


# In[ ]:


#%%
areas_lhs_its = genfromtxt('AM_LHS_ITS.csv', delimiter=',')
areas_lhs_its = areas_lhs_its[1:]
#%%
mean_lhs_its = np.mean(areas_lhs_its, axis = 1)
ci_its = calculate_confidence_interval(areas_lhs_its)
ci_its = ci_its[1:]


# In[ ]:


#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_lhs_its, label = 'Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, ci_its[:, :1].reshape(119), ci_its[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)


# In[ ]:


#%%
#%#%#%#%#%#%#%# comparing with pure random sampling #%#%#%#%#%#%#%#%#%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_lhs_its, label = 'LHS Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean2, label = 'Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, ci_its[:, :1].reshape(119), ci_its[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)
#%%


# In[ ]:


#%%
std_its = np.std(areas_lhs_its, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_its), label = 'Standard deviation')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.ylim(0.033, 0.0395)


# In[ ]:


# %%
#%#%#%#%#%#%#%#% comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#%#%#%#
std_its = np.std(areas_lhs_its, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_its), label = 'Standard deviation LHS')
ax.plot( num_iterations, (std_2), label = 'Standard deviation random sampling')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)


# In[ ]:


# %%
num_iterations = np.arange(100, 10000, 100, dtype = np.int16)
time_LHS = np.array([])
for i in tqdm(num_iterations):
    s = timer()
    area = mandel.compute_area_LHS(100, 1000, i)
    e = timer()
    time_LHS = np.append(time_LHS, (e-s))


# In[ ]:


#%%
fig, axs  = plt.subplots(figsize = (8,8))
axs.plot(num_iterations, time_LHS)
axs.plot(num_iterations, time)
plt.xlabel('number of iterations', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with random sampling', fontsize = 16)
