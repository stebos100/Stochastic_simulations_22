#%%
#importing the relevant libraries which will be used for this simulation 

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import random 
sns.set_theme()
from tqdm import tqdm
from numba import jit

#%%
"""going to firstly begin by investigating the various maps assocatied with the data, 
looking into the 51 cities nodes and thereafter the 280, then proceed to the 442
"""
#%%
def arrange_df_for_plotting(df):
    df.columns = ['x', 'y']
    df = df.assign(city=np.arange(1,df.shape[0] +1))
    return df

def graph_plotter_normal(df, name):
    fig, ax = plt.subplots(figsize=(12,12))
    p1 = sns.scatterplot(x="x", y="y", hue = "city", s = 150, data=df)

    if name == "51 cities":
        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+1, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='large', color='black')

    elif name == "280 cities":
        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+1, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='large', color='black') 

    elif name == "442 cities":
        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+1, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='medium', color='black')

    ax.set_xlabel('x-coordinate', fontsize =15)
    ax.set_ylabel('y-coordinate', fontsize = 15)
    ax.set_title('{} visualization'.format(name), fontsize = 16)
    plt.setp(p1.get_legend().get_texts(), fontsize='13') 
 
    # for legend title
    plt.setp(p1.get_legend().get_title(), fontsize='13')

#%%
##%#%#%#%#%#%#%#%#%#%#%##%#% loading the data and preparing it for plotting #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
#%#%#%#%#%#%#%#%#%#%#%#%#%#%
route_51 = pd.read_csv('TSP-Configurations/eil51.tsp.txt', delimiter=' ', index_col= 0, header = None)
route_51 = arrange_df_for_plotting(route_51)

route_280 = pd.read_csv('TSP-Configurations/a280.tsp.txt', delimiter=' ', index_col= 0, header = None)
route_280 = arrange_df_for_plotting(route_280)

route_442 = pd.read_csv('TSP-Configurations/pcb442.tsp.txt', delimiter=' ', index_col= 0, header = None)
route_442 = arrange_df_for_plotting(route_442)
#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#% plotting the data #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

graph_plotter_normal(route_51, "51 cities")
graph_plotter_normal(route_280, "280 cities")
graph_plotter_normal(route_442, "442 cities")

#%%
#%#%#%#%#%#%#%#%#%#%#%#%#%#% FUNCTIONS #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
def generate_distance_matrix(df):
    matrix = pd.DataFrame(data = np.zeros((len(df), len(df))), index = np.arange(1,len(df)+1),columns = np.arange(1,len(df)+1))
    for index, row_fixed in df.iterrows():
        for index2, row_iteration in df.iterrows():
            x_change = row_fixed["x"] - row_iteration["x"]
            y_change = row_fixed["y"] - row_iteration["y"]
            euclidean_distance = (x_change**2 + y_change**2)**0.5
            # print("the distance for this combination is: {}".format(euclidean_distance))
            matrix.loc[index, index2] = euclidean_distance
    return matrix


def distance(d_df, route):
    distance = 0
    for i in range(len(route)-1):
        distance += d_df.loc[route[i], route[i+1]]
    return distance


def generate_initial_path(df, if_replicable):

    if if_replicable == True:
        random.seed(10)
        random_route = [1] + [int(x) for x in random.sample(list(np.arange(2,(len(df.index)+1))), len(df.index)-1)] + [1]
    else: 
        random_route = [1] + [int(x) for x in random.sample(list(np.arange(2,(len(df.index)+1))), len(df.index)-1)] + [1]    
        
    return random_route

def swap(path, swap_first, swap_last):
    path_updated = np.concatenate((path[0:swap_first],
                                    path[swap_last:-len(path) + swap_first - 1:-1],
                                    path[swap_last + 1:len(path)]))

    return path_updated.tolist()

def swap_1(route):
    new_route = route[1:-1]
    pos_1, pos_2 = random.sample(range(0, len(new_route)), 2)
    new_route[pos_1], new_route[pos_2] = new_route[pos_2], new_route[pos_1]
    return [1] + new_route + [1]

def Temperature(n, a, b):
    T = a / np.log(n + b)
    return T

def fast_Temp(t, n):
    T = t / float(n + 1)
    return T
    
def get_best_schedule_and_distance(name, distance_matrix):
    dist = 0

    if name == "51":
        best_route = pd.read_csv('TSP-Configurations/eil51.opt.tour.txt', delimiter=' ', header = None)
        best_route.columns=['path']
        best = best_route["path"].values.tolist()

        dist = distance(distance_matrix, best)
        return best, dist

    elif name == "280":
        best_route = pd.read_csv('TSP-Configurations/a280.opt.tour.txt', delimiter=' ', header = None)
        best_route.columns=['path']
        best = best_route["path"].values.tolist()

        dist = distance(distance_matrix, best)
        return best, dist
    elif name == "442":
        best_route = pd.read_csv('TSP-Configurations/pcb442.opt.tour.txt', delimiter=' ', header = None)
        best_route.columns=['path']
        best = best_route["path"].values.tolist()

        dist = distance(distance_matrix, best)
        return best, dist

#%%
dx = generate_distance_matrix(route_442)
#%%
a,b = get_best_schedule_and_distance("442", dx)
b
#%%
def two_opt_2(Markov_chain_length, inital_route, distance_matrix,t0, stop):

    dist = distance(distance_matrix , inital_route)
    T = t0
    cost_record = [dist]
    routes = []
    temps =[]

    best_route_cost = dist
    best_route = inital_route
    c = 0
    for k in tqdm(range(stop)):
        for i in range(Markov_chain_length):
            for swap_first in range(1, len(inital_route) - 2):
                for swap_last in range(swap_first + 1, len(inital_route) - 1):

                    before_start = best_route[swap_first - 1]
                    start = best_route[swap_first]
                    end = best_route[swap_last]
                    after_end = best_route[swap_last+1]
                    before = distance_matrix[before_start][start] + distance_matrix[end][after_end]
                    after = distance_matrix[before_start][end] + distance_matrix[start][after_end]

                    new_route_temp = swap(best_route, swap_first, swap_last)
                    new_distance_temp = distance(distance_matrix, new_route_temp)
                    difference_in_cost = best_route_cost - new_distance_temp

                    if after < before:
                        best_route = new_route_temp
                        best_route_cost = new_distance_temp

                    else:
                        # Acceptance probability
                        P = min(np.exp(difference_in_cost/T), 1)
                        # Accept or not?
                        if random.uniform(0,1) < P:
                            best_route = new_route_temp
                            best_route_cost = new_distance_temp
        T = T - t0/stop
        c += 1
        cost_record.append(distance(distance_matrix, best_route))
        routes.append(best_route)

    return routes, best_route, cost_record
    
#%%
def two_opt(Markov_chain_length, inital_route, distance_matrix,t0, stop):

    dist = distance(distance_matrix , inital_route)
    T = t0
    cost_record = [dist]
    routes = []
    temps = []

    best_route_cost = dist
    best_route = inital_route
    k = 0
    for k in tqdm(range(stop)):
        k+=1
        for i in range(Markov_chain_length):
            # for swap_first in range(1, len(inital_route) - 2):
            #     for swap_last in range(swap_first + 1, len(inital_route) - 1):
                    new_route = swap_1(best_route)
                    new_distance = distance(distance_matrix, new_route)

                    difference_in_cost = best_route_cost - new_distance

                    if difference_in_cost > 0: 
                        best_route = new_route
                        best_route_cost = new_distance
                    else:
                        # Acceptance probability
                        P = min(np.exp(difference_in_cost/T), 1)
                        # Accept or not?
                        if random.uniform(0,1) < P:
                            best_route = new_route
                            best_route_cost = new_distance

        cost_record.append(distance(distance_matrix, best_route))
        routes.append(best_route)

        T = T - t0/stop
        temps.append(T)

    return routes, best_route, cost_record, temps
# %%
path = generate_initial_path(route_51, False)
path_2 = generate_initial_path(route_280, False)
dx = generate_distance_matrix(route_51)

#%%
routes, best_route, cost_record, temps = two_opt(100, path, dx, 5, 1000)
#%%

plt.plot(cost_record)

#%%
len(cost_record)
# %%
#%#%#%#%#%#%%#%#%#%%##%#%%#%# TO DO #%#%#%#%#%#%%#%#%#%#%#%#%#%#%%#
#%#%#%#%#%#%%#%#%#%%##%#%%#%# TO DO #%#%#%#%#%#%%#%#%#%#%#%#%#%#%%#
#%#%#%#%#%#%%#%#%#%%##%#%%#%# TO DO #%#%#%#%#%#%%#%#%#%#%#%#%#%#%%#
#%#%#%#%#%#%%#%#%#%%##%#%%#%# TO DO #%#%#%#%#%#%%#%#%#%#%#%#%#%#%%#

# 1.) 
#
#


# %%
