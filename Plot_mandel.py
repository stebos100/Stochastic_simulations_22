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
            if (z.real *z.real + z.imag *z.imag >= 2):
                return i
        return 255
    
    def convert_to_pixels_plot(self):
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
        z = 0
        for i in range(iterations):
            if np.abs(z) > 2:
                return False
            else:
                z = z*z + c
        return True

    def compute_area_random(self, num_runs, nsamples, numiterations):

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

    def return_area_matrix_constant_iterations(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_samples:
            area = self.compute_area_random(num_runs, i, num_iterations)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)

        return am

    def return_area_matrix_constant_samples(self, num_runs, num_samples, num_iterations, areas_matrix):
        am = areas_matrix
        for i in num_iterations:
            area = self.compute_area_random(num_runs, num_samples, i)
            area = np.array(area, dtype = np.float32)
            area = area.reshape(1,num_runs)
            am = np.concatenate((am, area), axis = 0)
        
        return am

    def calculate_required_samples(self, num_runs, num_iterations, d):

        samples = 900
        k = 100
        std = 0
        areas = 0
        while (d < k):
            area = area = self.compute_area_random(num_runs, samples, num_iterations)
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
            area = area = self.compute_area_random(num_runs, num_samples, num_iterations)
            area = np.array(area, dtype = np.float32)
            std_dev = np.std(area)
            std = std_dev
            num_runs += 500

        return samples, std



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
plt.show()
plt.savefig("mandel_plot.png", dpi = 700,  bbox_inches = 'tight', format = 'png' )
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
plt.savefig("mandel_plot_second.pdf", dpi = 900,  bbox_inches = 'tight', format = 'pdf' )


#%%

#%#%%#%#%#%#%#%#%#%#%#%#%#%#% Creating a second class instance for testing convergence #%#%#%#%#%#%#%#%#%#%#%#
RE_START = -2.0
RE_END = 1.0
IM_START = -1.2
IM_END = 1.2
image = np.zeros((10000 * 2,  10000* 2), dtype = np.uint32)
its = 1000

mandel= mandeL_plot(RE_START, RE_END, IM_START, IM_END, image, its)


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
#%%
samples_req = genfromtxt('num_samples.csv', delimiter=',')

#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[0,:], label ='number of samples required')
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Number of samples', fontsize = 14)
plt.title('Number of samples required for desired Standard deviation', fontsize = 16)
plt.legend(fontsize = 12)
# %%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[1,:])
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('sample standard deviation', fontsize = 14)
plt.title('Standard deviation for increasing iterations', fontsize = 16)
plt.legend(fontsize = 12)
plt.ylim(0.08, 0.105)
#%%
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(samples_req[3,:], samples_req[2,:])
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Sample area of MandelBrot set', fontsize = 14)
plt.title('Sample area for increasing iterations', fontsize = 16)
plt.legend(fontsize = 12)
ax.axhline(y = 1.5064, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.ylim(1.49, 1.52)
plt.legend(fontsize = 12)

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%%#%#% using the create area matrix function to store the data #%#%#%#%#%#%#%#%#%
num_runs = 1000
num_samples = np.arange(10, 12000, 100)
num_iterations = 2000
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% storing the data #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%%#
#%#%#%#%#%#%#%#%#%#%#%#%#%# I have edited this out, its already been done #%#%#%#%#%#%%#%#%#%#%
# import pandas as pd 
# areas1 = mandel.return_area_matrix(num_runs, num_samples, num_iterations, areas_matrix, 1)
# pd.DataFrame(areas).to_csv("area_matrix_one.csv")
# np.savetxt("AM_one.csv", areas, delimiter=",")

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# example of how to import the data #%#%#%#%#%#%#%#%#%#%#%
from numpy import genfromtxt
my_data = genfromtxt('AM_one.csv', delimiter=',')

#%%
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
#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%# plotting the standard devuation  of the resulst obtained #%#%#%#%#%#%#%#
std = np.std(my_data, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_samples, (std[1:]), label = 'Standard deviation')
plt.xlabel('Number of samples', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)

# %%
#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# conducting the second test #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
num_runs = 1000
num_samples = 2000
num_iterations = np.arange(100, 12000, 100)
areas_matrix = np.zeros(shape = (1, num_runs), dtype = np.float32)
#%%

#%#%#%%#%#%#%#%#%#%#%%#%#%#%#%#%%# convergence for increasing iterations #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#%#%
# areas2 = mandel.return_area_matrix_constant_samples(num_runs, num_samples, num_iterations, areas_matrix)
# import pandas as pd 
# pd.DataFrame(areas2).to_csv("area_matrix_two.csv")
# np.savetxt("AM_two.csv", areas2, delimiter=",")
# %%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%%# IMPORTING THE DATA #%#%#%#%#%#%#%#%#%#%#%#%%#%#%#%#%#
my_data2 = genfromtxt('AM_two.csv', delimiter=',')
#%#%#%#%#%#%#%#%#%#%#%#%#%#% generating the mean of all simulations along the rows #%#%#%#%#%#%
#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#% plotting the mean to the true solution to test the convergence of the solution #%#%#%#%#%#%#%#%#%#%#%#%#%#%#
mean2 = np.mean(my_data2, axis = 1)
mean2 = mean2[1:]
cis2 = calculate_confidence_interval(my_data2[1:])
cis2 = cis2[1:]
#%%
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(num_iterations, (mean2), label = 'Average Area for for 1000 runs, 2000 samples')
ax.axhline(y = 1.5064, color = 'r', linestyle = '--', label = 'True value of the area of the MandelBrot set')
plt.fill_between(num_iterations, cis2[:, :1].reshape(119), cis2[:, 1:].reshape(119),  color = 'blue', alpha = 0.15, label = '95% confidence interval')
plt.ylabel('Area of MandelBrot set', fontsize = 14)
plt.xlabel('Number of iterations', fontsize = 14)
plt.title('Convergent behaviour for increasing number of iterations', fontsize = 16)
plt.legend(fontsize = 12)
#%%
std = np.std(my_data2, axis = 1)
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize = (8,8))
ax.plot( num_iterations, (std[1:]), label = 'Standard deviation')
plt.ylim(0.05, 0.065)
plt.xlabel('Number of iterations', fontsize = 14)
plt.ylabel('Standard deviation of sample Area', fontsize = 14)
plt.title('Convergent behaviour of sample Standard deviation', fontsize = 16)
#%%