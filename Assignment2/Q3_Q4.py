#%%
#importing the various packages needed for the simulations 
import simpy
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy import integrate
import functions as Functions
plt.style.use("seaborn")
from numpy import genfromtxt

#%%         
class Setup_shortestjob(object):
    '''Class to setup Simpy environment for SPTF queues'''
    def __init__(self, env, arrival_rate, processing_capacity, server, waiting_times, n_samples, p_distribution):
        self.env = env
        self.arrival_rate = arrival_rate
        self.processing_capacity = processing_capacity
        self.server = server
        self.waiting_times = waiting_times
        self.n_samples_reached = env.event()
        self.n_samples = n_samples
        self.p_distribution = p_distribution
        self.action = env.process(self.run_shortest_task_first())


    def Shortest_task_first(self, processing_time, waiting_time):
        """task arrives, is served with a priority equal to the processing time and leaves."""
        arrive = self.env.now
        with self.server.request(priority=processing_time) as req:
            yield req
            waiting_time.append(env.now-arrive)
            yield self.env.timeout(processing_time)
        
    def run_shortest_task_first(self):
        while True:
            if  len(self.waiting_times)>self.n_samples:
                self.n_samples_reached.succeed()
                self.n_samples_reached = self.env.event()
            
            arrival_time = random.expovariate(self.arrival_rate)
            yield env.timeout(arrival_time)
            processing_time = self.p_distribution(self.processing_capacity)
            env.process(self.Shortest_task_first(processing_time, waiting_times))


def task(env, server, processing_time, waiting_time):
    """task arrives, is served and leaves."""
    arrive = env.now
    with server.request() as req:
        yield req
        waiting_time.append(env.now-arrive)
        yield env.timeout(processing_time)

class DES_MD_LT(object):
    def __init__(self, env, arrival_rate, service_rate, servers, waiting_times, numsamples, distribution):
        self.env = env
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server = servers
        self.waiting_times = waiting_times
        self.num_samples_count = env.event()
        self.n_samples = numsamples
        self.distribution = distribution
        self.action = env.process(self.run())

    def deterministic(self, x):
        '''Returns the deterministic time corresponding to a capacity of x'''
        return 1/x

    def longtail(self, x):
        '''Returns a long tail distribution with mean 1/x 
        where 25% has an exponential distribution with mean processing capacity = 5
        and 75% an exponential with a mean so that the mean of the distribution is 1/x '''
        mu_big = 5
        mu_small = 0.75*x*mu_big/(mu_big-0.25*x)
        a = random.random()
        if a < 0.25:
            n = random.expovariate(mu_big)
        else:
            n = random.expovariate(mu_small)
        return n


    def run(self):
        while True:
            if len(self.waiting_times) > self.n_samples:
                self.num_samples_count.succeed()
                self.num_samples_count = self.env.event()
            
            arrival_time = random.expovariate(self.arrival_rate)
            yield env.timeout(arrival_time)

            if self.distribution == 'longtail':
                service_time = self.longtail(self.service_rate)
                env.process(task(self.env, self.server, service_time, waiting_times))
            elif self.distribution == 'deterministic':
                service_time = self.deterministic(self.service_rate)
                env.process(task(self.env, self.server, service_time, waiting_times))
            else:
                print("the specified distribution does not comply")

#%%

"""using similiar paramters as the tests used for n =1,2,4 servers
"""
#%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%#
#%#%#%#%#%#%#%#%#%%#%#%# going to investigate shortest task first #%#%#%#%#%#%#%#%#%%#%#%##%#%#%
n_samples = 50000
n_servers = np.array([1])
steps = 10
arrival_rate = n_servers
p_min = 0.5
p_max = 0.995
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
waiting_times_mm1_shortest = np.zeros((1, steps, n_samples))
waiting_times_rho_s = np.zeros((1, steps,n_samples))

#%%
for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
        env = simpy.Environment()
        servers1 = simpy.PriorityResource(env, capacity=n_servers[i])
        waiting_times = []
        setup1 = Setup_shortestjob(env, arrival_rate[i], service_rate[j], servers1, waiting_times, n_samples,random.expovariate)
        env.run(until=setup1.n_samples_reached)
        waiting_times_rho_s[i, j, :] = setup1.waiting_times[:n_samples]
# np.save(f'results/waiting_times_rho_s', waiting_times_rho_s)
#%%
n_samples = 200000
n_servers = np.array([1,2,4])
steps = 20
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
runs = 25

waiting_times_SJF = np.zeros((3, steps, n_samples))
waiting_times_SJF_stacked = np.zeros((1, runs))
waiting_times_SJF_stacked.shape


#%%
# for i in range(len(n_servers)):
#     for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
#         waiting_times_SJF_stacked_temp = np.zeros((1, n_samples))
#         for k in range(runs):
#             env = simpy.Environment()
#             servers1 = simpy.PriorityResource(env, capacity=n_servers[i])
#             waiting_times = []
#             setup1 = Setup_shortestjob(env, arrival_rate[i], service_rate[j], servers1, waiting_times, n_samples,random.expovariate)
#             env.run(until=setup1.n_samples_reached)            
#             waiting_times_SJF_stacked_temp = np.vstack((waiting_times_SJF_stacked_temp, setup1.waiting_times[:n_samples]))   
#         appending = np.mean(waiting_times_SJF_stacked_temp[1:], axis = 1)
#         apend = appending.reshape(1, appending.shape[0])
#         waiting_times_SJF_stacked = np.vstack((waiting_times_SJF_stacked,apend))

# waiting_times_SJF_stacked = waiting_times_SJF_stacked[1:]
# np.savetxt("SJF_0.5_0.95.csv", waiting_times_SJF_stacked, delimiter=",")
#%%
waiting_times_SJF_stacked = genfromtxt('SJF_0.5_0.95.csv', delimiter=',')
relavant_std_MM = np.std(waiting_times_SJF_stacked, axis = 1)
relavant_means_MM = np.mean(waiting_times_SJF_stacked, axis = 1)

cis  = Functions.calculate_confidence_interval(waiting_times_SJF_stacked)
cis = cis[1:]
#%%
rho_range = np.linspace(p_min, p_max, 100)
mu_range = 1 / rho_range
theory = []

for rho, mu in zip(rho_range, mu_range):
    theory.append(Functions.SPTF(rho, mu, 1))
#%%
fig, ax1 = plt.subplots(figsize = (7,7))
fig2, ax2 = plt.subplots(figsize = (7,7))

ax1.plot(rho_range , theory, linestyle = '--', linewidth = 1, label = "Theoretical result for {} server".format(1))

for i in n_servers:
    if i == 1:
        ax1.scatter(p_range, relavant_means_MM[0:i*20], marker = "^", label = 'simulation values for {} server'.format(i))
        ax1.fill_between(p_range, relavant_means_MM[0:i*20] + 0.1, relavant_means_MM[0:i*20] - 0.1, alpha = 0.3, label = "0.05 width tolerance")
        ax2.plot(p_range, relavant_std_MM[0:i*20], marker = "^", label = 'standard deviation for {} server(s)'.format(i), linewidth = 2, markersize = 8)
    elif i == 2:
        ax1.plot(p_range, relavant_means_MM[((i)*10):((i+2)*10)], label = 'simulation values for {} server(s)'.format(i))
        ax2.plot(p_range, relavant_std_MM[((i)*10):((i+2)*10)], marker = "o", linestyle = "--", label = 'standard deviation for {} server(s)'.format(i), linewidth = 2, markersize = 8)

    elif i == 4:
        ax1.plot(p_range, relavant_means_MM[((i)*10):((i+2)*10)], label = 'simulation values for {} server(s)'.format(i))
        ax2.plot(p_range, relavant_std_MM[((i)*10):((i+2)*10)], marker = "8", linestyle  = "-.", label = 'standard deviation for {} server(s)'.format(i), linewidth = 2, markersize = 8)


ax1.legend(fontsize = 13)
ax1.set_xlabel( "Utilization rate " r'$\rho$', fontsize = 15)
ax1.set_ylabel(r'$\bar{W}$', fontsize = 15)
ax1.set_title("Waiting times for SJFS/1 - SJFS/n Queuing simulation", fontsize  = 15)
ax1.tick_params(axis='both', which='major', labelsize=13)


ax2.legend(fontsize = 13)
ax2.set_xlabel("Utilization rate " r'$\rho$', fontsize = 15)
ax2.set_ylabel(r'$S[\bar{W}]$', fontsize = 15)
ax2.set_title("Standard deviation for SJFS/1 - SJFS/n Queuing simulation", fontsize  = 15)
ax2.tick_params(axis='both', which='major', labelsize=12)
#%%
#%% 
#%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%
#%#%#%#%#%#%#%#%#%%#%#%# investigating deterministic distributions #%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%#
#global variables
# %%
n_samples = 200000
n_servers = np.array([1,2,4])
steps = 20
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
runs = 25

waiting_times_MDN_stacked = np.zeros((1, runs))

#%%
# for i in range(len(n_servers)):
#     for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]} and step {j}'):
#         waiting_times_MDN_stacked_temp = np.zeros((1, n_samples))
#         for k in range(runs):
#             env = simpy.Environment()
#             servers3 = simpy.PriorityResource(env, capacity=n_servers[i])
#             waiting_times = []
#             setup3 = DES_MD_LT(env, arrival_rate[i], service_rate[j], servers3, waiting_times, n_samples,'deterministic')
#             env.run(until=setup3.num_samples_count)
#             waiting_times_MDN_stacked_temp = np.vstack((waiting_times_MDN_stacked_temp, setup3.waiting_times[:n_samples]))   
#         appending = np.mean(waiting_times_MDN_stacked_temp[1:], axis = 1)
#         apend = appending.reshape(1, appending.shape[0])
#         waiting_times_MDN_stacked = np.vstack((waiting_times_MDN_stacked,apend))

# waiting_times_MDN_stacked = waiting_times_MDN_stacked[1:]
# np.savetxt("MDN_0.5_0.95.csv", waiting_times_MDN_stacked, delimiter=",")

#%%
waiting_times_MDN_stacked = genfromtxt("MDN_0.5_0.95.csv", delimiter=',')

relavant_std_MD = np.std(waiting_times_MDN_stacked, axis = 1)
relavant_means_MD = np.mean(waiting_times_MDN_stacked, axis = 1)

#%%
rho_range = np.linspace(p_min, p_max, 1000)
mu_range = 1 / rho_range
theoretical_det = np.zeros((1,len(rho_range)))
for i in n_servers:
    values = []
    for rho, mu in zip(rho_range, mu_range):
        values.append(Functions.MDn(rho, mu, i))

    theoretical_det = np.vstack((theoretical_det, values))
    
theoretical_det = theoretical_det[1:]
#%%

fig5, ax5 = plt.subplots(figsize = (8,8))

l1 = ax5.scatter(p_range, relavant_means_MD[0:20], marker = "^", s = 80, c ='teal')
m1 = ax5.plot(rho_range , theoretical_det[0, :], linestyle = '--', linewidth = 1.5, c = 'teal')

l2 = ax5.scatter(p_range, relavant_means_MD[20:40], s = 80, c = 'm')
m2 = ax5.plot(rho_range , theoretical_det[1, :], linestyle = '--', linewidth = 1.5, c = 'm')

l3 = ax5.scatter(p_range, relavant_means_MD[40:60], s = 80, c  = 'chocolate')
m3 = ax5.plot(rho_range , theoretical_det[2, :], linestyle = '--', linewidth = 1.5, c = 'chocolate')

p1 = ax5.fill_between(p_range, relavant_means_MD[0:20] + 0.1, relavant_means_MD[0:20] - 0.1, alpha = 0.4, color = "mediumpurple")
p2 = ax5.fill_between(p_range, relavant_means_MD[20:40] + 0.1, relavant_means_MD[20:40] - 0.1, alpha = 0.4, color = "mediumpurple")
p3 = ax5.fill_between(p_range, relavant_means_MD[40:60] + 0.1, relavant_means_MD[40:60]- 0.1, alpha = 0.4, color = "mediumpurple")


ax5.legend(fontsize = 13)
ax5.set_xlabel( "Utilization rate [" r'$\rho$'"]", fontsize = 15)
ax5.set_ylabel(r'$\bar{W}$', fontsize = 16)
ax5.set_title("Waiting times for M/D/1 - M/D/n Queueing simulation", fontsize  = 15)
ax5.tick_params(axis='both', which='major', labelsize=13)
ax5.legend(["simulation values for 1 server", "Theoretical result for 1 server", \
   "simulation values for 2 server(s)" , "Theoretical result for 2 server(s)", "simulation values for 4 server(s)", \
       "Theoretical result for 4 server(s)", "0.05 std deviation tolerance value"], fontsize =13)
fig5.savefig('M_D_N_QUEUE.png', bbox_inches='tight', dpi = 600 )


fig6, ax6 = plt.subplots(figsize = (8,8))

ax6.plot(p_range, relavant_std_MD[0:20], marker = "^", label = 'standard deviation for 1 server(s)', linewidth = 2, markersize = 8, c = 'teal')
ax6.plot(p_range, relavant_std_MD[20:40], marker = "o", linestyle = "--", label = 'standard deviation for 2 server(s)', linewidth = 2, markersize = 8, c = 'm')
ax6.plot(p_range, relavant_std_MD[40:60], marker = "8", label = 'standard deviation for 4 server(s)', linewidth = 2, markersize = 8, c = 'chocolate')


ax6.legend(fontsize = 13)
ax6.set_xlabel("Utilization rate [" r'$\rho$'"]", fontsize = 15)
ax6.set_ylabel(r'$S[\bar{W}]$', fontsize = 15)
ax6.set_title("Standard deviation for M/D/1 - M/D/n Queue simulation", fontsize  = 15)
ax6.tick_params(axis='both', which='major', labelsize=13)
fig6.savefig('M_D_N_QUEUE_STD.png', bbox_inches='tight', dpi = 600 )

#%%
#%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%#% going to investigate long tailed distribution #%#%#%#%#%#%#%#%#%%#%%#

#%%
n_samples = 200000
n_servers = np.array([1,2,4])
steps = 20
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
runs = 25

waiting_times_MLN_stacked = np.zeros((1, runs))

#%%
# for i in range(len(n_servers)):
#     for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]} and step {j}'):
#         waiting_times_MLN_stacked_temp = np.zeros((1, n_samples))
#         for k in range(runs):
#             env = simpy.Environment()
#             servers2 = simpy.PriorityResource(env, capacity=n_servers[i])
#             waiting_times = []
#             setup2 = DES_MD_LT(env, arrival_rate[i], service_rate[j], servers2, waiting_times, n_samples,'longtail')
#             env.run(until=setup2.num_samples_count)
#             waiting_times_MLN_stacked_temp = np.vstack((waiting_times_MLN_stacked_temp, setup2.waiting_times[:n_samples]))   
#         appending = np.mean(waiting_times_MLN_stacked_temp[1:], axis = 1)
#         apend = appending.reshape(1, appending.shape[0])
#         waiting_times_MLN_stacked = np.vstack((waiting_times_MLN_stacked,apend))

# waiting_times_MLN_stacked = waiting_times_MLN_stacked[1:]
# np.savetxt("MLN_0.5_0.95.csv", waiting_times_MLN_stacked, delimiter=",")
#%%

waiting_times_MLN_stacked = genfromtxt('MLN_0.5_0.95.csv', delimiter=',')
waiting_times_MLN_stacked.shape
#%%
relavant_std_ML = np.std(waiting_times_MLN_stacked, axis = 1)
relavant_means_ML = np.mean(waiting_times_MLN_stacked, axis = 1)

#%%
rho_range = np.linspace(p_min, p_max, 1000)
mu_range = 1 / rho_range
theoretical_longtail = np.zeros((1,len(rho_range)))
for i in n_servers:
    values = []
    for rho, mu in zip(rho_range, mu_range):
        values.append(Functions.longtail_pred(rho, mu, i))

    theoretical_longtail = np.vstack((theoretical_longtail, values))
    
theoretical_longtail = theoretical_longtail[1:]

#%%

fig3, ax3 = plt.subplots(figsize = (8,8))

l1 = ax3.scatter(p_range, relavant_means_ML[0:20], marker = "^", s = 80)
m1 = ax3.plot(rho_range , theoretical_longtail[0, :], linestyle = '--', linewidth = 1.5)

l2 = ax3.scatter(p_range, relavant_means_ML[20:40], s = 80)
m2 = ax3.plot(rho_range , theoretical_longtail[1, :], linestyle = '--', linewidth = 1.5)

l3 = ax3.scatter(p_range, relavant_means_ML[40:60], s = 80)
m3 = ax3.plot(rho_range , theoretical_longtail[2, :], linestyle = '--', linewidth = 1.5)

p1 = ax3.fill_between(p_range, relavant_means_ML[0:20] + 0.2, relavant_means_ML[0:20] - 0.2, alpha = 0.4, color = "mediumpurple")
p2 = ax3.fill_between(p_range, relavant_means_ML[20:40] + 0.2, relavant_means_ML[20:40] - 0.2, alpha = 0.4, color = "mediumpurple")
p3 = ax3.fill_between(p_range, relavant_means_ML[40:60] + 0.2, relavant_means_ML[40:60]- 0.2, alpha = 0.4, color = "mediumpurple")


ax3.legend(fontsize = 13)
ax3.set_xlabel( "Utilization rate [" r'$\rho$'"]", fontsize = 15)
ax3.set_ylabel(r'$\bar{W}$', fontsize = 15)
ax3.set_title("Waiting times for M/L/1 - M/L/n Queueing simulation", fontsize  = 16)
ax3.tick_params(axis='both', which='major', labelsize=13)
ax3.legend(["simulation values for 1 server", "Theoretical result for 1 server", \
   "simulation values for 2 server(s)" , "Theoretical result for 2 server(s)", "simulation values for 4 server(s)", \
       "Theoretical result for 4 server(s)", "0.05 std deviation tolerance value"], fontsize =13)

fig3.savefig('M_L_N_QUEUE.png', bbox_inches='tight', dpi = 600)

fig4, ax4 = plt.subplots(figsize = (8,8))

ax4.plot(p_range, relavant_std_ML[0:20], marker = "^", label = 'standard deviation for 1 server(s)', linewidth = 2, markersize = 8)
ax4.plot(p_range, relavant_std_ML[20:40], marker = "o", linestyle = "--", label = 'standard deviation for 2 server(s)', linewidth = 2, markersize = 8)
ax4.plot(p_range, relavant_std_ML[40:60], marker = "8", label = 'standard deviation for 4 server(s)', linewidth = 2, markersize = 8)


ax4.legend(fontsize = 13)
ax4.set_xlabel("Utilization rate [" r'$\rho$'"]", fontsize = 15)
ax4.set_ylabel(r'$S[\bar{W}]$', fontsize = 15)
ax4.set_title("Standard deviation for M/L/1 - M/L/n Queue simulation", fontsize  = 16)
ax4.tick_params(axis='both', which='major', labelsize=13)
fig4.savefig('M_L_N_QUEUE_STD.png', bbox_inches='tight', dpi = 600 )

# %%
