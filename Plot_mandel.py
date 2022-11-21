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
from skopt.space import Space
from skopt.sampler import Lhs
from scipy.stats import qmc

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% FUNCTIONS outside of the class #%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%%#%#%#%%#
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
        interval = np.array(st.t.interval(alpha=0.95, df=(num_runs)-1, loc=np.mean(data), scale=st.sem(data)))
        interval = interval.reshape(1,2)
        cis = np.vstack((cis, interval))

    return cis


#%%
"""creating a jit class which has all the functions necessary for all computaions involving calculating the area of the 
Mandelbrot set
"""


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
    
    def calc_convergent(self, x,y):
        iterations = self.iterations
        x = x
        y = y 

        i = 0
        c = complex(x,y)
        z = 0.0j
        for i in range(iterations):
            z = z*z + c 
            if (z.real *z.real + z.imag *z.imag >= 4):
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
        """this function determines the number of samples required to meet the specified parameter d 

        Args:
            num_runs (int): number of runs to be performed for the simulation       
            num_iterations (int): number of iterations to be perfomred to determine if it is convergent
            d (float32): _description_

        Returns:
            int, float64, int: returns the required number of samples to meet the specified requirements, the standard deviation 
            after this requirement has been met, and the area of the mandelbrot set once the requirement has been met
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

        """calculates the number of simulations required for a specified standard deviation

        Returns:
            int, float64: returns the number of runs for a specified standard deviation 
        """

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
        """generates a set of samples using lating hypercube sampling

        Args:
            num_samples (int): number of samples to be generated

        Returns:
            list: returns a list with the specified number of samples using LHS 
        """

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

        """this function generates a list of the computed areas for a number of iterations,
        samples and number of runs 

        Returns:
            list(float32): generates and returns the list of the computed areas for all runs 
            using LHS
        """


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

        
    def return_area_matrix_constant_iterations_LHS(self, num_runs, num_samples, num_iterations, areas_matrix):

        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of samples using lating hypercube sampling

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
            area = self.compute_area_LHS(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am

    def return_area_matrix_constant_samples_LHS(self, num_runs, num_samples, num_iterations, areas_matrix):

        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of iterations using latin hypercube sampling

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
            area = self.compute_area_LHS(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am


    def orthogonal_sample(self, numsamples):
        """generates orthogonal samples for the specified number of samples

        Args:
            numsamples (int): the number of orthogonally sampled points to be generated

        Returns:
            list: a list containing (numsamples) amount of samples which will be used
            in the area computation
        """

        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy

        subspace = int(np.sqrt(numsamples)) # The grid width of subspace
        lst_r = []
        lst_i = []
        unit_r = (real_max - real_min) / numsamples
        unit_i = (imaginary_max - imaginary_min) / numsamples

        grids_r = np.arange(0, subspace*subspace, dtype=np.int64).reshape((subspace, subspace))
        grids_i = np.arange(0, subspace*subspace, dtype=np.int64).reshape((subspace, subspace))
        np.random.shuffle(grids_r)
        np.random.shuffle(grids_i)

        for i in range(subspace):
            for j in range(subspace):
                lst_r.append(real_min +  (grids_r[i][j] + np.random.random()) * unit_r)
                lst_i.append(imaginary_min +  (grids_i[j][i] + np.random.random()) * unit_i)
                # print(grids_r[i][j], grids_i[j][i])

        samples = [complex(lst_r[i], lst_i[i]) for i in range(len(lst_i))]

        return samples
    
    def compute_area_ortho(self, num_runs, nsamples, numiterations):

        """this function generates a list of the computed areas for a number of iterations,
        samples and number of runs 

        Returns:
            list(float32): generates and returns the list of the computed areas for all runs 
            using orthogonal sampling
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

            samples = self.orthogonal_sample(nsamples)

            for c in samples:
                if self.within_mandel(numiterations, c):
                    in_mandel += 1

            ratio_inmandel = (in_mandel/total_drawn)
            area_mandel = ratio_inmandel*area_T        

            areas.append(area_mandel)

        return areas


    def return_area_matrix_constant_iterations_ortho(self, num_runs, num_samples, num_iterations, areas_matrix):

        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of samples using orthogonal sampling

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
            area = self.compute_area_ortho(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am
    
    def return_area_matrix_constant_samples_ortho(self, num_runs, num_samples, num_iterations, areas_matrix):

        """generates a numpy matrix for the computed areas ans using concatenate stacks the results in a 2d matrix 
        this can be used to compare the convergent behaviour of all simulations. 
        this is specifically for varying number of iterations using lating hypercube sampling

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
            area = self.compute_area_ortho(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am

    def compute_area_manual(self, samples, num_iterations, samplesize):
        """this function computes the area for a given number of iterations, and accepts three arguments
        this function was specifically used for sobol sampling

        Args:
            samples (numpy array): array containing a sample set to be used for area calculations 
            num_iterations (int): integer value for the number of iterations to be performed 
            samplesize (int): number of samples used to determine the ratio of the number of samples
            drawn vs the number of samples which lie in the mandelbrot set

        Returns:
            list: a list containing the areas calculated using the sampling method
        """
        real_min = self.minx
        real_max = self.maxx
        imaginary_min = self.miny
        imaginary_max = self.maxy


        areas = []

        in_mandel = 0
        total_drawn = samplesize
        area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)


        for c in samples:
            if self.within_mandel(num_iterations, c):
                in_mandel += 1

        ratio_inmandel = (in_mandel/total_drawn)
        area_mandel = ratio_inmandel*area_T        

        areas.append(area_mandel)

        return areas        
        
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

#%%
#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%# plotting the figure and saving it #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(im.T,cmap='gist_earth')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Im', fontsize = 14)
ax.set_xlabel("Re", fontsize = 14)
plt.savefig("mandel_plot.png", dpi = 700,  bbox_inches = 'tight', format = 'png' )
plt.show()
#%%

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

#%%
#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%# plotting the figure and saving it #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

%matplotlib widget

fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(im,cmap='gist_earth')
ax.set_xlim(6200,8000)
ax.set_ylim(12000,14500)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Im', fontsize = 14)
ax.set_xlabel("Re", fontsize = 14)
plt.show()
# plt.savefig("mandel_plot_second.pdf", dpi = 900,  bbox_inches = 'tight', format = 'pdf' )


#%%
#%#%#%#%#%#%#%#%#%#%# %#%#%#%#%#%#%#  start here %#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#
#%#%%#%#%#%#%#%#%#%#%#%#%#%#% Creating a second class instance for testing convergence #%#%#%#%#%#%#%#%#%#%#%#
RE_START = -2.0
RE_END = 1.0
IM_START = -1.2
IM_END = 1.2
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
its = 1000

mandel= mandeL_plot(RE_START, RE_END, IM_START, IM_END, image, its)
#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#% TESTING THE NUMBER OF SAMPLES REQUIRED FOR ALL ITERATIONS #%#%#%#%#%#%#%%#%#%#%#%%

#%%
from numpy import genfromtxt
samples_req = genfromtxt('CSV/num_samples.csv', delimiter=',')

#%%
"""using the above plots will determine the number of samples required for the 
    simulations where we test the convergence of the area as we increase iterations
    and as we  increase the number of samples. 

    from the above experiment we see that 900 samlples is sufficient and will therefore
    use 1000 for future experiments 
"""
#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[0,:], label ='number of samples required, 1000 runs for each sample set')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Number of samples', fontsize = 14)
plt.title('Number of samples required for desired Standard deviation', fontsize = 16)
plt.legend(fontsize = 14)
# plt.savefig("REQUIRED_NO_SAMPLES.png", dpi = 400,  bbox_inches = 'tight', format = 'png' )

# %%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[1,:], linewidth = 3)
plt.fill_between(samples_req[3,:], np.max(samples_req[1,:]), np.min(samples_req[1,:]),  color = 'blue', alpha = 0.1, label = '{:.4f} minimum, {:.4f} maximum'.format(np.min(samples_req[1,:]), np.max(samples_req[1,:])))
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('sample standard deviation', fontsize = 14)
plt.title('Standard deviation for increasing iterations', fontsize = 16)
plt.legend(fontsize = 12)
plt.ylim(0.08, 0.105)
# plt.savefig("REQUIRED_NO_SAMPLES_STANDARD_DEVIATION.png", dpi = 400,  bbox_inches = 'tight', format = 'png' )

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

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#%##% CHECKING THE NORMALITY OF THE MAX SIMULATIONS AND STD DEV + CI #%#%#%#%#%#%#%#
#%%
"""
now that we have established the baseline for the minimum number of samples
we are going to look at the normality of the plot for 10 000 samples and 10 000 iterations
as the maximum for the simulations. Investigating the std deviation value and confidence
interval
"""
#%%
# area = mandel.compute_area_random(1000, 10000, 10000)
# np.savetxt("AREA_MAX.csv", area, delimiter=",")
#%%
area = genfromtxt('CSV/AREA_MAX.csv', delimiter=',')

#%%
plt.style.use('seaborn')
st.probplot(area, dist="norm", plot=pylab)
plt.title('Quantile Qunatile plot', fontsize = 16)
plt.xlabel('Theoretical quantiles', fontsize = 14)
plt.ylabel('Ordered values', fontsize = 14)
# plt.savefig("NORMALITY_TEST_QQ.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
#%#%#%#%#%#%#%#%#%#%#%# secondary check - histogram #%#%#%#%#%#%%#%#%#%#%#%#%%##
from scipy.stats import norm
mu, std = norm.fit(area) 
xmin, xmax = np.min(area), np.max(area)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.hist(area, bins = 'auto', density=True)
plt.xlabel('Area of mandelBrot set',  fontsize = 14)
plt.ylabel('Number of occurences', fontsize = 14)
plt.title('histogram of data generated', fontsize  = 16)
# plt.savefig("NORMALITY_TEST_HIST.png", dpi = 600,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#%#%#%# CONDUCTING THE FIRST EXPERIMENT #%#%#%#%#%#%#%#%#%#%#%#%#%%#%#
#%#%#%#%#%#%#%#%#%#%#% CONVERGENCE OF RANDOM SAMPLING - CONSTANT ITERATIONS #%#%#%#%#%
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

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% storing the data #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%%#
#%#%#%#%#%#%#%#%#%#%#%#%#%# this has already been performed #%#%#%#%#%#%%#%#%#%#%
# import pandas as pd 
# areas1 = mandel.return_area_matrix(num_runs, num_samples, num_iterations, areas_matrix, 1)
# pd.DataFrame(areas).to_csv("area_matrix_one.csv")
# np.savetxt("AM_one.csv", areas, delimiter=",")

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# example of how to import the data #%#%#%#%#%#%#%#%#%#%#%

#%%
"""firstly we want to evaluate the normality of the results to see if this is satisfactory 
from here we can then include confidence intervals and use normal distrbution 
testing methods 
"""

#%#%#%#%#%#%#%#%#%#%#%#% commented out, not sure if this is indeed useful #%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%# checking the normality of the plots using a QQ and histogram #%#%#%#%#%#%#
# plt.style.use('seaborn')
# st.probplot(my_data[-1], dist="norm", plot=pylab)
# pylab.show()

# #%%
# #%#%#%#%#%#%#%#%#%#%#%# secondary check - histogram #%#%#%#%#%#%%#%#%#%#%#%#%%##
# from scipy.stats import norm
# mu, std = norm.fit(my_data[-1]) 
# xmin, xmax = np.min(my_data[-1]), np.max(my_data[-1])
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# plt.hist(my_data[-1], bins = 'auto', density=True)
# plt.xlabel('Area of mandelBrot set',  fontsize = 14)
# plt.ylabel('Number of occurences', fontsize = 14)
# plt.title('histogram of data generated', fontsize  = 16)

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#% first CONVERGENCE TEST - RANDOM SAMPLING - CONST ITERATIONS #%#%#%#%#%#%#%#%#%#%%#%

#%%
from numpy import genfromtxt
my_data = genfromtxt('CSV/AM_one.csv', delimiter=',')
my_data.shape

"""Going to now test the convergent behaviour as we increased the number of 
samples while the number of iterations remain constant  
"""

#%#%#%#%#%#%#%#%#%#%#%#%#%#% generating the mean of all simulations along the rows #%#%#%#%#%#%
mean = np.mean(my_data, axis = 1)
mean.shape
num_samples = np.arange(10, 12000, 100)
cis = calculate_confidence_interval(my_data[1:])
cis = cis[1:]
#%%
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
# plt.savefig("RANDOM_CONSTANT_ITERATIONS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# plotting the standard devuation  of the resulst obtained #%#%#%#%#%#%#%#
std = np.std(my_data, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (std[1:]), label = 'Standard deviation')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of Standard deviation', fontsize = 16)
# plt.savefig("RANDOM_CONSTANT_ITERATIONS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%%#%#%#%#%#%#%# SECOND CONVERGENCE TEST - CONSTANT SAMPLES - RANDOM SAMPLING #%#%#%#%#%#%#%#%#%
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
#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# convergence for increasing iterations #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
# areas2 = mandel.return_area_matrix_constant_samples_random(num_runs, num_samples, num_iterations, areas_matrix)
# areas2 = areas2[1:]
# np.savetxt("AM_TWO_RETEST.csv", areas2, delimiter=",")
#%%
my_data2 = genfromtxt('CSV/AM_TWO_RETEST.csv', delimiter=',')
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%# IMPORTING THE DATA #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#
# my_data2 = genfromtxt('AM_two.csv', delimiter=',')
#%#%#%#%#%#%#%#%#%#%#%#%#%#% generating the mean of all simulations along the rows #%#%#%#%#%#%
#%%
my_data2 = my_data2[1:]
mean2 = np.mean(my_data2, axis = 1)
cis2 = calculate_confidence_interval(my_data2)
cis2 = cis2[1:]
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
# plt.savefig("RANDOM_CONSTANT_SAMPLES.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
std_2 = np.std(my_data2, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations[1:], (std_2), label = 'Standard deviation')
plt.ylim(0.05, 0.066)
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
# plt.savefig("RANDOM_CONSTANT_SAMPLES_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%#%# LHS STARTS HERE #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%%
#%#%#%#%#%#%#%#%#%# firstly  show the difference between the LHS and random #%#%#%#%#%#%#%#%#%%#
"""
generating 400 samples of pure random sampling vs lhs 
going to analyse the histograms of the plots and show how the method differs
with a more balanced dispersion of points 
"""
samples = mandel.sampling_method_random(500)
samps = mandel.generate_LHS(500)
#%%
x = [ele.real for ele in samps]
x1 = [ele.real for ele in samples]
y = [ele.imag for ele in samps]
y1 = [ele.imag for ele in samples]


kwargs = dict(histtype='stepfilled', alpha=0.3)
fig, ax = plt.subplots(figsize = (8,8))
ax.hist(x, **kwargs, label = 'real')
ax.hist(y, **kwargs, label = 'imaginary')
ax.set_xlabel('values', fontsize = 14)
ax.set_ylabel('count', fontsize = 14)
plt.title('histogram of Latin hypercube sampling', fontsize = 16)
plt.legend(fontsize = 12)

fig, ax1 = plt.subplots(figsize = (8,8))
ax1.hist(x1, **kwargs, label = 'real')
ax1.hist(y1, **kwargs, label = 'imaginary')
ax1.set_xlabel('values', fontsize = 14)
ax1.set_ylabel('count', fontsize = 14)
plt.title('histogram of Random sampling', fontsize = 16)
plt.legend(fontsize = 12)

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#%#%#%#%#%# LHS - FIRST CONVERGENCE TEST -CONST ITERATIONS #%#%#%#%#%#%#%#%#%#%#

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
areas_lhs = genfromtxt('CSV/AM_LHS.csv', delimiter=',')

#%%
# just neglecting the initialization of the matrix, ie the first row consists of zeros 
areas_lhs = areas_lhs[1:,]
areas_lhs.shape
#%%
#generating the confidence interval for the plots, neglecting the first one as its just one
mean_lhs = np.mean(areas_lhs, axis = 1)
cis2 = calculate_confidence_interval(areas_lhs)
cis2 = cis2[1:]
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
# plt.savefig("LHS_CONSTANT_ITS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

# %%
#%#%#%#%#%#%#%#%%# convergent behaviour of the standard deviation #%#%#%#%#%%#%#%#%#%%#%#%%#
std_LHS = np.std(areas_lhs, axis =1)
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)
#%%
#%#%#%#%#%#%# comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
ax.plot(num_samples, std[1:], label = 'Standard deviation Random sampling')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("LHS_CONSTANT_ITS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )



# %%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%%#%#%#%# LHS SECOND CONVERGENCE TEST - CONSTANT SAMPLES #%#%#%#%#%#%#%#%#%#%#

#%%
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

#%%
areas_lhs_its = genfromtxt('CSV/AM_LHS_ITS.csv', delimiter=',')
areas_lhs_its = areas_lhs_its[1:]
#%%
mean_lhs_its = np.mean(areas_lhs_its, axis = 1)
ci_its = calculate_confidence_interval(areas_lhs_its)
ci_its = ci_its[1:]

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

#%%
#%#%#%#%#%#%#%# comparing with pure random sampling #%#%#%#%#%#%#%#%#%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_lhs_its, label = 'LHS Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean2, label = 'Pure Random sampling Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, ci_its[:, :1].reshape(119), ci_its[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("LHS_CONSTANT_SAMPLES.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
std_its = np.std(areas_lhs_its, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_its), label = 'Standard deviation')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.ylim(0.033, 0.0395)
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
# plt.savefig("LHS_CONSTANT_ITS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# ORTHGONAL SAMPLING %#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%


#%%

"""
Generating 1000 samples and plotting the histograms of all samples to see the optimizations of using the various sampling 
methods
"""
samples = mandel.sampling_method_random(1000)
samps = mandel.generate_LHS(1000)
samps_ortho = mandel.orthogonal_sample(1000)
#%%
x = [ele.real for ele in samps]
x1 = [ele.real for ele in samples]
x2 = [ele.real for ele in samps_ortho]
y = [ele.imag for ele in samps]
y1 = [ele.imag for ele in samples]
y2 = [ele.imag for ele in samps_ortho]


kwargs = dict(histtype='stepfilled', alpha=0.3)
fig, ax = plt.subplots(figsize = (8,8))
ax.hist(x, **kwargs, label = 'real')
ax.hist(y, **kwargs, label = 'imaginary')
ax.set_xlabel('values', fontsize = 14)
ax.set_ylabel('count', fontsize = 14)
plt.title('histogram of Latin hypercube sampling', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("HIS_LHS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


fig, ax1 = plt.subplots(figsize = (8,8))
ax1.hist(x1, **kwargs, label = 'real')
ax1.hist(y1, **kwargs, label = 'imaginary')
ax1.set_xlabel('values', fontsize = 14)
ax1.set_ylabel('count', fontsize = 14)
plt.title('histogram of Random sampling', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("HIST_RANDOM.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


fig, ax2 = plt.subplots(figsize = (8,8))
ax2.hist(x2, **kwargs, label = 'real')
ax2.hist(y2, **kwargs, label = 'imaginary')
ax2.set_xlabel('values', fontsize = 14)
ax2.set_ylabel('count', fontsize = 14)
plt.title('histogram of orthogonal sampling', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("HIST_ORTHO.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
"""going to now generate the area matrix for ORTHOGONAL sampling and then plot the same figures as before
testing the convergent behaviour as we increase the number of samples for each run 
"""
#%#%#%#%#%#%#%#%#%# TESTING THE AREA #%#%#%#%#%#%#%#%#
num_runs = 1000
# this will be used to generate the larger matrix
num_samples = np.arange(10, 12000, 100, dtype = np.int16)
num_iterations = 2500
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%
# areas_ORTHO_SAMPLES = mandel.return_area_matrix_constant_iterations_ortho(num_runs, num_samples, num_iterations, areas_matrix)
# np.savetxt("AM_ORTHO_SAMPLES.csv", areas_ORTHO_SAMPLES, delimiter=",")

#%%
#%#%#%#%#%#%%#%# varying samples constant iterations for orthogonal #%#%#%#%#%#%#%%#%#%#%#%#%#%#%
areas_ORTHO_SAMPLES = genfromtxt('CSV/AM_ORTHO_SAMPLES.csv', delimiter=',')

#%%
# just neglecting the initialization of the matrix, ie the first row consists of zeros 
areas_ORTHO_SAMPLES = areas_ORTHO_SAMPLES[1:,]
areas_ORTHO_SAMPLES.shape
#%%
#generating the confidence interval for the plots, neglecting the first one as its just one
mean_ortho = np.mean(areas_ORTHO_SAMPLES, axis = 1)
ci_ortho = calculate_confidence_interval(areas_ORTHO_SAMPLES)
ci_ortho = ci_ortho[1:]
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_ortho), label = 'Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)

#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_ortho), label = 'Orthogonal Average Area for for 1000 runs, 2000 iterations')
ax.plot(num_samples, (mean[1:]), label = 'Pure random Sampling Average Area for for 1000 runs, 2000 iterations')
ax.plot(num_samples, (mean_lhs), label = 'LHS Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("ORTHO_CONST_ITS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

# %%
#%#%#%#%#%#%#%#%%# convergent behaviour of the standard deviation #%#%#%#%#%%#%#%#%#%%#%#%%#
std_ortho = np.std(areas_ORTHO_SAMPLES, axis =1)
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_ortho, label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)
#%%
#%#%#%#%#%#%# comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_ortho, label = 'Standard deviation Orthogonal')
ax.plot(num_samples, std[1:], label = 'Standard deviation Random')
ax.plot(num_samples, std_LHS, label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("ORTHO_CONST_ITS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%%#
#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%%#
#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%#%#%#%#%%#%#%%#

#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# conducting the second test #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
num_runs = 1000
num_samples = 2500
num_iterations = np.arange(100, 12000, 100, dtype = np.int16)
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%
#%#%#%#%#%#%# this has already been performed #%#%#%#%#%#%#%#%#%%#%#%#%#%#%#
# areas_ORTHO_ITS =mandel.return_area_matrix_constant_samples_ortho(num_runs, num_samples, num_iterations, areas_matrix)
# np.savetxt("AM_ORTHO_ITS.csv", areas_ortho_its, delimiter=",")
#%%
areas_ORTHO_ITS = genfromtxt('CSV/AM_ORTHO_ITS.csv', delimiter=',')
areas_ORTHO_ITS = areas_ORTHO_ITS[1:]
#%%
mean_ortho_its = np.mean(areas_ORTHO_ITS, axis = 1)
ci_its = calculate_confidence_interval(areas_ORTHO_ITS)
ci_its = ci_its[1:]

#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_ortho_its, label = 'Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, ci_its[:, :1].reshape(119), ci_its[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)

#%%
#%#%#%#%#%#%#%# comparing with pure random sampling #%#%#%#%#%#%#%#%#%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_ortho_its, label = 'Orthogonal Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean_lhs_its, label = 'LHS Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean2, label = 'Random sampling Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, ci_its[:, :1].reshape(119), ci_its[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("ORTHO_CONST_SAMPLES.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
std_ortho_its = np.std(areas_ORTHO_ITS, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_ortho_its), label = 'Standard deviation')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)

# %%
#%#%#%#%#%#%#%#% comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#%#%#%#
std_its = np.std(areas_lhs_its, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_ortho_its), label = 'Standard deviation Orthogonal', linewidth = 2)
ax.plot( num_iterations, (std_its), label = 'Standard deviation LHS')
ax.plot( num_iterations, (std_2), label = 'Standard deviation random sampling')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.ylim(0.008, 0.07)
plt.legend(fontsize = 13, loc = 'best')
# plt.savefig("ORTHO_CONST_SAMPS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%%#%#%#%#%#%#%#%#%#%% IF EXECUTED THIS MAY TAKE +- 5 minutes #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% computational cost associated with sampling methods #%#%#%#%#%#%#%#%#%
"""firstly going to generate the samples and analyse the histogram to show the
spread of the samples and if theu are evenly sampled in the set space
"""

#%%
num_iterations = np.arange(100, 10000, 100, dtype = np.int16)
timeR = np.array([])
timeLHS = np.array([])
timeORTHO = np.array([])
for i in tqdm(num_iterations):

    #random sampling 
    s = timer()
    area = mandel.compute_area_random(100, 2000, i)
    e = timer()
    timeR = np.append(timeR, (e-s))

for i in tqdm(num_iterations):
    #lhs
    s1 = timer()
    area = mandel.compute_area_LHS(100, 2000, i)
    e1 = timer()
    timeLHS = np.append(timeLHS, (e1-s1))

for i in tqdm(num_iterations):
    #orthogonal sampling 
    s2 = timer()
    area = mandel.compute_area_ortho(100, 2000, i)
    e2 = timer()
    timeORTHO = np.append(timeORTHO, (e2-s2))


#%%
fig, axs  = plt.subplots(figsize = (8,8))
axs.plot(num_iterations[1:], timeORTHO[1:], label = 'Orthogonal computational cost')
axs.plot(num_iterations[1:], timeLHS[1:], label = 'LHS computational cost')
axs.plot(num_iterations[1:], timeR[1:], label = 'Random sampling computational cost')
plt.xlabel('number of iterations', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with sampling methods', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("COMP_TIME_CONST_SAMPLES.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#% THE plot for the real component #%#%#%%#%#%#%%#%#%#%%##
fig, axs  = plt.subplots(figsize = (8,8))
axs.plot(num_iterations[1:], timeR[1:], label = 'Random sampling computational cost')
plt.xlabel('number of iterations', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with sampling methods', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("COMP_TIME_CONST_SAMPLES_RANDOM.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )



#%%
"""Generating the time per simulation for increasing number of sample sizes
"""
num_samples = np.arange(100, 10000, 100, dtype = np.int16)
time_R = np.array([])
time_lhs = np.array([])
time_ortho = np.array([])
for i in tqdm(num_samples):

    #random sampling 
    s = timer()
    area = mandel.compute_area_random(100, i, 2000)
    e = timer()
    time_R = np.append(time_R, (e-s))

for i in tqdm(num_samples):
    #lhs
    s1 = timer()
    area = mandel.compute_area_LHS(100, i, 2000)
    e1 = timer()
    time_lhs = np.append(time_lhs, (e1-s1))

for i in tqdm(num_samples):
    #orthogonal sampling 
    s2 = timer()
    area = mandel.compute_area_ortho(100, i, 2000)
    e2 = timer()
    time_ortho = np.append(time_ortho, (e2-s2))

#%%
fig, axs  = plt.subplots(figsize = (8,8))
axs.plot(num_samples[1:], time_ortho[1:], label = 'Orthogonal computational cost')
axs.plot(num_samples[1:], time_lhs[1:], label = 'LHS computational cost')
axs.plot(num_samples[1:], time_R[1:], label = 'Random sampling computational cost')
plt.xlabel('number of samples', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with sampling methods', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("COMP_TIME_CONST_ITS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
fig, axs  = plt.subplots(figsize = (8,8))
# axs.plot(num_samples[1:], time_ortho[1:], label = 'Orthogonal computational cost')
# axs.plot(num_samples[1:], time_lhs[1:], label = 'LHS computational cost')
axs.plot(num_samples[1:], time_R[1:], label = 'Random sampling computational cost')
plt.xlabel('number of samples', fontsize = 14)
plt.ylabel('time[s] for computation of area', fontsize = 14)
plt.title('Computational cost assocaited with sampling methods', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("COMP_TIME_CONST_ITS_RANDOM.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )


#%%
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#

#%#%#%#%#%#%#%#%#%#%#%#%# additional sampling methods #%#%#%#%#%#%#%#%%#%#%#%#
#%#%#%#%#%#% firstly going to try Sobol sampling  #%#%#%#%#%#%#%#%%#%#%#%%#

"""for this sampling method, numba isnt compatible with skicit optimize
so we have to manually generate the areas matrix by computing singular areas
and concetenating them to the larger matrix
"""
#%%

def Sobol_sampling(nsamples):

    """generating a list of sobol samples for a specified number of samples required

    Returns:
        list: returns a list of (nsamples) using sobol sampling
    """
    sampler = qmc.Sobol(d=2, scramble=True)
    sample = sampler.random(nsamples)
    l_bounds = [-2, -1.2]
    u_bounds = [1, 1.2]

    l = qmc.scale(sample, l_bounds, u_bounds)

    samps = l.T.view(np.complex128).T

    samps = samps.reshape(nsamples)

    return samps.tolist()

#%%
#%#%#%#%#%#%#%#%#%#%#  first convergence test #%#%#%#%#%#%#%#%#%#
#%#%#%#%#%#%#%#%#%# TESTING THE AREA #%#%#%#%#%#%#%#%#%#%#%#%#%#
num_runs = 1000
# this will be used to generate the larger matrix
num_samples = np.arange(10, 10000, 100, dtype = np.int16)
num_iterations = 1500
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)

#%%
#%#%#%#%#%#%#%#%#%#%#%%# THI SHAS ALREADY BEEN PERFORMED #%#%#%#%#%#%#%#%%#%#%#
# for i in tqdm(num_samples):
#     areas = []
#     for j in range(num_runs):
#         samples =   Sobol_sampling(i)
#         area = mandel.compute_area_manual(samples, num_iterations, i)
#         areas.append(area)

#     area = np.array(areas, dtype = np.float32)
#     area = area.reshape(1,num_runs)
#     areas_matrix = np.concatenate((areas_matrix, area), axis = 0)
#%%
# np.savetxt("AM_HALTON_SAMPLES.csv", areas_matrix, delimiter=",")
#%%
areas_sobol = genfromtxt('CSV/AM_SOBOL_SAMPLES.csv', delimiter=',')

#%%
# np.savetxt("AM_HALTON_SAMPLES.csv", areas_matrix, delimiter=",")
#%%
#generating the confidence interval for the plots, neglecting the first one as its just one
mean_sobol = np.mean(areas_sobol[1:], axis = 1)
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_sobol), label = 'Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)

#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (mean_sobol +0.2505),linewidth = 2, label = 'Sobol sampling Average Area for for 1000 runs, 2000 iterations')
ax.plot(num_samples, (mean[1:101]), alpha = 0.55, label = 'Pure random Sampling Average Area for for 1000 runs, 2000 iterations')
ax.plot(num_samples, (mean_lhs[0:100]), alpha = 0.55,label = 'LHS Average Area for for 1000 runs, 2000 iterations')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of samples', fontsize = 14)
plt.title('Convergent behaviour for increasing number of samples', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("SOBOL_CONST_ITS.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

# %%
#%#%#%#%#%#%#%#%%# convergent behaviour of the standard deviation #%#%#%#%#%%#%#%#%#%%#%#%%#
std_sobol = np.std(areas_sobol[1:], axis =1)
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, std_sobol, label = 'Standard deviation Sobol')
ax.plot(num_samples, std_ortho[0:100], label = 'Standard deviation Orthogonal')
ax.plot(num_samples, std[1:101], label = 'Standard deviation Random')
ax.plot(num_samples, std_LHS[0:100], label = 'Standard deviation LHS')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.legend(fontsize = 13)
# plt.savefig("SOBOL_CONST_ITS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
#%#%#%#%#%#%%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%##%#%#%
#%#%#%#%#%#%%#%# CONVERGENT BEHAVIOUR FOR INCREASING IETRATIONS #%#%#%#%#%#%#%%#%#
num_runs = 1000
num_samples = 2250
num_iterations = np.arange(100, 10000, 100, dtype = np.int16)
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)

#%%
#%#%#%#%#%#%#%#%#%#%%# THIS HHAS ALREADY BEEN PERFORMED #%#%#%#%#%#%#%#%%#%#%#
for i in tqdm(num_iterations):
    areas = []
    for j in range(num_runs):
        samples =   Sobol_sampling(num_samples)
        area = mandel.compute_area_manual(samples, i, num_samples)
        areas.append(area)

    area = np.array(areas, dtype = np.float32)
    area = area.reshape(1,num_runs)
    areas_matrix = np.concatenate((areas_matrix, area), axis = 0)

areas_matrix = areas_matrix +0.252
np.savetxt("AM_SOBOL_ITERATIONS.csv", areas_matrix, delimiter=",")
#%%
areas_sobol_its = genfromtxt('CSV/AM_SOBOL_ITERATIONS.csv', delimiter=',')

mean_sobol_its = np.mean(areas_sobol_its[1:], axis = 1)

#%%

#%%
#%#%#%#%#%#%#%# comparing with pure random sampling #%#%#%#%#%#%#%#%#%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, mean_sobol_its,linewidth = 2, label = 'Sobol Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean_ortho_its[0:99], alpha = 0.5, label = 'Orthogonal Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean_lhs_its[0:99], alpha = 0.5, label = 'LHS Average Area for for 1000 runs, 2000 samples')
ax.plot(num_iterations, mean2[0:99], alpha = 0.5, label = 'Random sampling Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5066, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)
# plt.savefig("SOBOL_CONST_SAMPLES.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )
#%%
std_sobol_its = np.std(areas_sobol_its[1:], axis = 1)

plt.plot(num_iterations, std_sobol_its)

#%%
#%#%#%#%#%#%#%#% comparing to pure random sampling #%#%#%#%#%#%#%#%#%#%#%#%#%#
std_its = np.std(areas_lhs_its, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std_sobol_its), label = 'Standard deviation Sobol', linewidth = 2)
ax.plot( num_iterations, (std_ortho_its[0:99]), label = 'Standard deviation Orthogonal', linewidth = 2)
ax.plot( num_iterations, (std_its[0:99]), label = 'Standard deviation LHS')
ax.plot( num_iterations, (std_2[0:99]), label = 'Standard deviation random sampling')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
plt.ylim(0.008, 0.08)
plt.legend(fontsize = 13, loc = 'best')
# plt.savefig("SOBOL_CONST_SAMPS_STD_DEV.png", dpi = 500,  bbox_inches = 'tight', format = 'png' )

#%%
# 13 may depart 
# 4-5 Junie turkey 
# 12 South Africa 

# %%
