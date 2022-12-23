
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
from statistics import mean

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

    if name == "442 cities":
        fig, ax = plt.subplots(figsize=(13,12))
        p1 = sns.scatterplot(x="x", y="y", hue = "city", s = 200, data=df)

        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+3, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='x-large', color='black')


        ax.set_xlabel('x-coordinate', fontsize =24)
        ax.set_ylabel('y-coordinate', fontsize = 24)
        ax.set_title('{} visualization'.format(name), fontsize = 24)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.setp(p1.get_legend().get_texts(), fontsize='18') 
        plt.setp(p1.get_legend().get_title(), fontsize='18') 

    elif name == "280 cities":
        fig, ax = plt.subplots(figsize=(12,12))
        p1 = sns.scatterplot(x="x", y="y", hue = "city", s = 200, data=df)

        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+2, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='x-large', color='black') 

        ax.set_xlabel('x-coordinate', fontsize =20)
        ax.set_ylabel('y-coordinate', fontsize = 20)
        ax.set_title('{} visualization'.format(name), fontsize = 22)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.setp(p1.get_legend().get_texts(), fontsize='17') 
        plt.setp(p1.get_legend().get_title(), fontsize='17')


    elif name == "51 cities":
        fig, ax = plt.subplots(figsize=(8,8))
        p1 = sns.scatterplot(x="x", y="y", hue = "city", s = 200, data=df)

        for line in range(1,df.shape[0] +1):
            p1.text(df.x[line]+2, df.y[line], 
            df.city[line], horizontalalignment='left', 
            size='large', color='black')

        ax.set_xlabel('x-coordinate', fontsize =16)
        ax.set_ylabel('y-coordinate', fontsize = 16)
        ax.set_title('{} visualization'.format(name), fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.setp(p1.get_legend().get_texts(), fontsize='15') 
        plt.setp(p1.get_legend().get_title(), fontsize='15') 

    return fig

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

five_one = graph_plotter_normal(route_51, "51 cities")
# plt.savefig("51_cities_config.png", bbox_inches = "tight", dpi = 350)

#%%
two_eighty = graph_plotter_normal(route_280, "280 cities")
# plt.savefig("280_cities_config.png", bbox_inches = "tight", dpi = 350)

#%%
four_four_two = graph_plotter_normal(route_442, "442 cities")
# plt.savefig("442_cities_config.png", bbox_inches = "tight", dpi = 350)

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

def cooling_schedule(k, cooling, T0):
    if cooling == "exponential":
        alpha = 0.8
        T = T0 *(alpha**k)     #0.8 < alpha < 0.9
    
    elif cooling == "linear":
        alpha = 0.5 
        T = T0 / (1+alpha*k)  #alpha > 0

    elif cooling == "logarithmic":
        alpha = 1.5
        T = T0 / (1 + alpha * np.log(1+k)) #alpha > 1
        
    elif cooling == "quadratic":
        alpha = 0.5
        T = T0 / (1 + (alpha*k**2))  #alpha > 0
        
    else:
        return ("Please give a valid cooling schedule")
        
    return T

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
#%%
def two_opt_2(Markov_chain_length, inital_route, distance_matrix,t0, stop, cooling):

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
        T = cooling_schedule(k, cooling, t0)
        temps.append(T)
        c += 1
        cost_record.append(distance(distance_matrix, best_route))
        routes.append(best_route)

    return routes, best_route, cost_record
    
#%%
def two_opt(Markov_chain_length, inital_route, distance_matrix,t0, stop, cooling):

    dist = distance(distance_matrix , inital_route)
    T = t0
    cost_record = [dist]
    routes = [inital_route]
    temps = [t0]

    best_route_cost = dist
    best_route = inital_route
    k = 0
    probs = [0]
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
                        probs.append(P)
                        # Accept or not?
                        if random.uniform(0,1) < P:
                            best_route = new_route
                            best_route_cost = new_distance

        cost_record.append(distance(distance_matrix, best_route))
        routes.append(best_route)

        if cooling == None:
            T = T 
            temps.append(T)
        elif cooling == "normal":
            T = T - t0/stop
            temps.append(T)
        else:
            T = cooling_schedule(k, cooling, t0)
            temps.append(T)

    return routes, best_route, cost_record, temps, probs
# %%
path1 = generate_initial_path(route_51, False)
dx1 = generate_distance_matrix(route_51)

path_2 = generate_initial_path(route_280, False)
dx_2 = generate_distance_matrix(route_280)

path_3 = generate_initial_path(route_442, False)
dx_3 = generate_distance_matrix(route_442)

#%%
b1,b11 = get_best_schedule_and_distance("51", dx1)
b2, b22 = get_best_schedule_and_distance("280", dx_2)
b3, b33 = get_best_schedule_and_distance("442", dx_3)

#%%
from numpy import genfromtxt
d_442 = genfromtxt("Cost_record_MC_10K_1000_ITS_442")
#%%
fig, ax = plt.subplots(figsize = (6,6))
ax.plot(d_442)
ax.set_ylabel("Distance", fontsize = 14)
ax.set_xlabel("Number of iterations", fontsize = 14)
ax.set_title("Optimum distance for 442 city configuration",  fontsize = 16)
ax.set_xticklabels([0,0,150,300, 450, 600, 750, 900, 1050, 1200])
plt.legend([d_442[-1]], fontsize = 13)
# plt.savefig("442_OPT.png", bboxinches = 'tight', dpi = 350)

#%%

d_280 = genfromtxt("Cost_record_MC_10K_1000_ITS")

fig, ax = plt.subplots(figsize = (6,6))
ax.plot(d_442)
ax.set_ylabel("Distance", fontsize = 14)
ax.set_xlabel("Number of iterations", fontsize = 14)
ax.set_title("Optimum distance for 280 city configuration",  fontsize = 16)
ax.set_xticklabels([0,0,150,300, 450, 600, 750, 900, 1050, 1200])
plt.legend([d_280[-1]], fontsize = 13)
# plt.savefig("442_OPT.png", bboxinches = 'tight', dpi = 350)
#%%
markov_chain_length = 100
num_iterations = 2000
T0 = [10, 50, 100, 500, 1000,2000, 3000]
cooling = "exponential"
initial = np.arange(0, num_iterations +1)
results = pd.DataFrame()
results["number"] = initial
probabilities = []
#%%
# for i in range(len(T0)):
#     routes, best_route, cost_record, temps, probs = two_opt(markov_chain_length, path_2, dx_2, T0[i], num_iterations, cooling)
#     probabilities.append(mean(probs))

#     data = {
#    'routes {}'.format(T0[i]): routes,
#    'cost record {}'.format(T0[i]): cost_record,
#    'temperatures {}'.format(T0[i]): temps
#     }

#     dataframe = pd.DataFrame.from_dict(data)

#     results = pd.concat([results, dataframe], axis=1)

# results.to_csv("EXP_mc500_T0.csv", index=False)
#%%
results_EXP = pd.read_csv("cooling_schedules_data/EXP_mc500_T0.csv")
#%%
fig, ax = plt.subplots(figsize = (6,6))
sns.set_theme(style="darkgrid")
for i in range(len(T0)):
    s = sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_EXP, label = "Inital temperature {}".format(T0[i]))

ax.set_xlabel('number of iterations', fontsize = 15)
ax.set_ylabel('Distance of shortest path', fontsize = 15)
ax.set_title("Influence of T0 for Eponential cooling", fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.legend(fontsize = 13)
# fig.savefig("expo_convergence.png", bbox_inches = 'tight', dpi = 350)

#%%
markov_chain_length = 100
num_iterations = 2000
T0 = [10, 50, 100, 500, 1000,2000, 3000]
cooling = "logarithmic"
initial = np.arange(0, num_iterations +1)
results = pd.DataFrame()
results["number"] = initial
probs_log = []
#%%
# for i in range(len(T0)):
#     routes, best_route, cost_record, temps, probs = two_opt(markov_chain_length, path_2, dx_2, T0[i], num_iterations, cooling )
#     probs_log.append(mean(probs))

#     data = {
#    'routes {}'.format(T0[i]): routes,
#    'cost record {}'.format(T0[i]): cost_record,
#    'temperatures {}'.format(T0[i]): temps
#     }

#     dataframe = pd.DataFrame.from_dict(data)

#     results = pd.concat([results, dataframe], axis=1)

# results.to_csv("LOG_mc500_T0.csv", index=False)
#%%
results_LOG = pd.read_csv("cooling_schedules_data/LOG_mc500_T0.csv")

fig, ax = plt.subplots(figsize = (6,6))
sns.set_theme(style="darkgrid")

for i in range(len(T0)):
    s = sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_LOG, label = "Inital temperature {}".format(T0[i]))

ax.set_xlabel('number of iterations', fontsize = 15)
ax.set_ylabel('Distance of shortest path', fontsize = 15)
ax.set_title("Influence of T0 for logarithmic cooling", fontsize = 17)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize = 13, loc = "center right")
# fig.savefig("LOG_convergence.png", bbox_inches = 'tight', dpi = 350)

#%%
markov_chain_length = 100
num_iterations = 2000
T0 = [10, 50, 100, 500, 1000,2000, 3000]
cooling = "quadratic"
initial = np.arange(0, num_iterations +1)
results = pd.DataFrame()
results["number"] = initial
probs_log = []

#%%
# for i in range(len(T0)):
#     routes, best_route, cost_record, temps, probs = two_opt(markov_chain_length, path_2, dx_2, T0[i], num_iterations, cooling)

#     data = {
#    'routes {}'.format(T0[i]): routes,
#    'cost record {}'.format(T0[i]): cost_record,
#    'temperatures {}'.format(T0[i]): temps
#     }

#     dataframe = pd.DataFrame.from_dict(data)

#     results = pd.concat([results, dataframe], axis=1)

# results.to_csv("QUAD_mc500_T0.csv", index=False)
#%%
results_QUAD = pd.read_csv("cooling_schedules_data/QUAD_mc500_T0.csv")

fig, ax = plt.subplots(figsize = (6,6))
sns.set_theme(style="darkgrid")

for i in range(len(T0)):
    s = sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_QUAD, label = "Inital temperature {}".format(T0[i]))


ax.set_xlabel('number of iterations', fontsize = 15)
ax.set_ylabel('Distance of shortest path', fontsize = 15)
ax.set_title("Influence of T0 for Quadratic cooling", fontsize = 17)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize = 13, loc = "center right")
# fig.savefig("QUAD_convergence.png", bbox_inches = 'tight', dpi = 350)
#%%

markov_chain_length = 100
num_iterations = 2000
T0 = [10, 50, 100, 500, 1000,2000, 3000]
cooling = "linear"
initial = np.arange(0, num_iterations +1)
results = pd.DataFrame()
results["number"] = initial
probs_log = []
#%%
# for i in range(len(T0)):
#     routes, best_route, cost_record, temps, probs = two_opt(markov_chain_length, path_2, dx_2, T0[i], num_iterations, cooling)

#     data = {
#    'routes {}'.format(T0[i]): routes,
#    'cost record {}'.format(T0[i]): cost_record,
#    'temperatures {}'.format(T0[i]): temps
#     }

#     dataframe = pd.DataFrame.from_dict(data)

#     results = pd.concat([results, dataframe], axis=1)

# results.to_csv("QUAD_mc500_T0.csv", index=False)

#%%
results_LINEAR = pd.read_csv("cooling_schedules_data/QUAD_mc500_T0.csv")

fig, ax = plt.subplots(figsize = (6,6))
sns.set_theme(style="darkgrid")

for i in range(len(T0)):
    s = sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_LINEAR, label = "Inital temperature {}".format(T0[i]))

ax.set_xlabel('number of iterations', fontsize = 15)
ax.set_ylabel('Distance of shortest path', fontsize = 15)
ax.set_title("Influence of T0 for Linear cooling", fontsize = 17)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize = 13, loc = "center right")
# fig.savefig("LINEAR_convergence.png", bbox_inches = 'tight', dpi = 350)
#%%

sns.set_theme(style="darkgrid")
for i in range(len(T0)):
    # sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_QUAD, label = "{}".format(T0[i]))
    sns.lineplot(x="number", y='cost record {}'.format(T0[i]), data=results_LINEAR, label = "{}".format(T0[i]))

# %%
from statistics import mean
markov_chain_length = 50
num_iterations = np.arange(10, 110, 10)
num_iterations_2 = np.arange(100, 600, 100)
num_its = np.concatenate((num_iterations, num_iterations_2), axis = 0)
T0 = 50
cooling_schedules = ["exponential", "linear", "logarithmic", "quadratic"]
sims = 10

#%%

# for j in cooling_schedules:
#     array = np.ones(shape = (1, num_its.shape[0]))
#     print("starting with {} cooling".format(j))
#     vals = []
#     for i in tqdm(num_its):
#         print("starting simulation {}".format(i))
#         cost_r = []
#         for k in range(sims):
#             routes, best_route, cost_record, temps = two_opt(markov_chain_length, path_2, dx_2, 50, i, j )
#             cost_r.append(cost_record[-1])

#         avg = mean(cost_r)
#         vals.append(avg)
#     values = np.array(vals)
#     array = np.vstack((array, values))
#     np.savetxt("{}_cooling_ITS.csv".format(j), array, delimiter=",")
# %%
markov_chain_length = [10, 50, 100, 250, 500, 750, 1000]
num_iterations = [10, 50, 100, 250, 500, 750, 1000]
cooling_scheds = ['quadratic']
T0 = 166
#%%
# for k in cooling_scheds:
#     print(k)
#     chain_heatmap = np.zeros((len(markov_chain_length), len(num_iterations)))
#     all_results_heatmap = []
#     for i in range(len(markov_chain_length)):
#         for j in range(len(num_iterations)):
#             a,b,c, d, e = two_opt(markov_chain_length[i], path_2, dx_2, T0, num_iterations[j],k)
#             chain_heatmap[i,j] = c[-1]
#             all_results_heatmap.append(c)

#     np.savetxt("{}_markovchain_vs_its_heatmap_T0=166".format(k), chain_heatmap)
# %%
from numpy import genfromtxt
exponential_heatmap = genfromtxt('cooling_schedules_data/exponential_markovchain_vs_its_heatmap_T0=166')
logarithmic_heatmap = genfromtxt('cooling_schedules_data/logarithmic_markovchain_vs_its_heatmap_T0=166')
quadratic_heatmap = genfromtxt('cooling_schedules_data/quadratic_markovchain_vs_its_heatmap_T0=166')

#%%
fig, ax = plt.subplots(figsize = (7,6))
ax = sns.heatmap(exponential_heatmap, cmap = 'viridis')
ax.invert_yaxis()
ax.set_xlabel("Iterations", fontsize = 16)
ax.set_ylabel("Markov chain lenght", fontsize = 16)
ax.set_xticklabels(num_iterations, fontsize = 13)
ax.set_yticklabels(num_iterations, fontsize = 13)
ax.set_title("exponential ccooling schedule minimum eucledian distance", fontsize = 17)
# plt.savefig("EXPO_Heatmap_route_280_T0=70.png", bbox_inches = 'tight', dpi = 350)
plt.show()
# %%
fig, ax = plt.subplots(figsize = (7,6))
ax = sns.heatmap(logarithmic_heatmap, cmap='viridis')
ax.invert_yaxis()
ax.set_xlabel("Iterations", fontsize = 16)
ax.set_ylabel("Markov chain lenght", fontsize = 16)
ax.set_xticklabels(num_iterations, fontsize = 13)
ax.set_yticklabels(num_iterations, fontsize = 13)
ax.set_title("Logartihmic cooling schedule  minimum eucledian distance", fontsize = 16)
# plt.savefig("LOG_Heatmap_route_280_T0=70.png", bbox_inches = 'tight', dpi = 350)
plt.show()
#%%
fig, ax = plt.subplots(figsize = (7,6))
ax = sns.heatmap(quadratic_heatmap, cmap='viridis')
ax.invert_yaxis()
ax.set_xlabel("Iterations", fontsize = 16)
ax.set_ylabel("Markov chain lenght", fontsize = 16)
ax.set_xticklabels(num_iterations, fontsize = 13)
ax.set_yticklabels(num_iterations, fontsize = 13)
ax.set_title("quadratic cooling schedule minimum eucledian distance", fontsize = 16)
# plt.savefig("QUAD_Heatmap_route_280_T0=70.png", bbox_inches = 'tight', dpi = 350)
plt.show()
# %%
