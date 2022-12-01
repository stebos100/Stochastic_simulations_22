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
steps = 20
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
mean_waiting_times = np.zeros((len(n_servers), steps))

for i in range(len(n_servers)):
    for j in range(steps):
        mean_waiting_times[i, j] = np.mean(waiting_times_rho_s[i, j, :])


mean_waiting_times.shape
# %%
rho_range = np.linspace(p_min, p_max, 1000)
mu_range = 1 / rho_range
theory = []

for rho, mu in zip(rho_range, mu_range):
    theory.append(Functions.SPTF(rho, mu, 1))

#%%
for i in range(len(n_servers)):
    plt.scatter(p_range, mean_waiting_times[i], marker = 'X', s=100, label = '%s servers' % n_servers[i])
plt.plot(rho_range , theory, linestyle = '--', linewidth = 1, label = "Theoretical result for {} server".format(i))
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()


#%%
for i in range(len(n_servers)):
    plt.plot(p_range, mean_waiting_times[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()
    
# %%
for i in range(len(n_servers)):
    plt.plot(service_rate, mean_waiting_times[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()

#%% 
#%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%#
#%#%#%#%#%#%#%#%#%#%#%#%#% going to investigate long tailed distribution #%#%#%#%#%#%#%#%#%%#%%#
n_samples = 10
n_servers = np.array([1,2,4])
steps = 10
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
waiting_times_MDN = np.zeros((3, steps, n_samples))
waiting_times_MDN_stacked = np.zeros((1, 10))
waiting_times_MDN_stacked.shape
#%%

for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
        env = simpy.Environment()
        servers2 = simpy.PriorityResource(env, capacity=n_servers[i])
        waiting_times = []
        setup2 = DES_MD_LT(env, arrival_rate[i], service_rate[j], servers2, waiting_times, n_samples,'longtail')
        env.run(until=setup2.num_samples_count)
        print(setup2.waiting_times[:n_samples])
        waiting_times_MDN[i, j, :] = setup2.waiting_times[:n_samples]


waiting_times_MDN.shape

#%%
n_samples = 20000
n_servers = np.array([1])
steps = 10
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
waiting_times_MDN_stacked = np.zeros((1, 100))
runs = 100

#%%
for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]} and step {j}'):
        waiting_times_MDN_stacked_temp = np.zeros((1, n_samples))
        for k in range(runs):
            env = simpy.Environment()
            servers2 = simpy.PriorityResource(env, capacity=n_servers[i])
            waiting_times = []
            setup2 = DES_MD_LT(env, arrival_rate[i], service_rate[j], servers2, waiting_times, n_samples,'longtail')
            env.run(until=setup2.num_samples_count)
            waiting_times_MDN_stacked_temp = np.vstack((waiting_times_MDN_stacked_temp, setup2.waiting_times[:n_samples]))   
        appending = np.mean(waiting_times_MDN_stacked_temp[1:], axis = 1)
        apend = appending.reshape(1, appending.shape[0])
        waiting_times_MDN_stacked = np.vstack((waiting_times_MDN_stacked,apend))

waiting_times_MDN_stacked = waiting_times_MDN_stacked[1:]
#%%
relavant_std = np.std(waiting_times_MDN_stacked, axis = 1)
relavant_means = np.mean(waiting_times_MDN_stacked, axis = 1)

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
theoretical_longtail.shape
#%%

fig, ax = plt.subplots(figsize = (8,8))
ax.scatter(p_range, relavant_means, marker = "^")
ax.fill_between(p_range, relavant_means - relavant_std, relavant_means + relavant_std, alpha = 0.3)
ax.plot(rho_range, theoretical_longtail[0,:,])

#%%
mean_waiting_times_LT = np.zeros((len(n_servers), steps))

for i in range(len(n_servers)):
    for j in range(steps):
        mean_waiting_times_LT[i, j] = np.mean(waiting_times_MDN[i, j, :])


# np.savetxt("mean_waiting_time_0.5-0.95_LT.csv", mean_waiting_times_LT, delimiter=",")
mean_waiting_times_LT.shape
# %%
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
# mean_waiting_times_LT = genfromtxt('mean_waiting_time_0.5-0.95_LT.csv', delimiter=',')

for i in range(len(n_servers)):
    plt.scatter(p_range, mean_waiting_times_LT[i], marker='^',label = '%s servers' % n_servers[i], linewidth = 2)
colors = ['cornflowerblue', 'forestgreen', 'lightcoral']
for i in range(3):
    plt.plot(rho_range, theoretical_longtail[i, :], linestyle = '--', color = colors[i] ,alpha = 0.9, label = 'theoretical value for {} servers'.format(i + 1))
plt.xlabel(r'$\rho$', fontsize = 17)
plt.ylabel("Waiting time", fontsize = 14)
plt.title("Waiting times for M/L/n vs amount of servers available", fontsize  = 15)
plt.legend(fontsize = 13)
    
# %%
for i in range(len(n_servers)):
    plt.plot(service_rate, mean_waiting_times_LT[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()
    
# %%
#%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%
#%#%#%#%#%#%#%#%#%%#%#%# investigating deterministic distributions #%#%#%#%#%#%#%#%#%%#%#%##%#%#%#%#%#%#%#%#%%#%#%#
#global variables
n_samples = 500000
n_servers = np.array([1,2,4])
steps = 10
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)
waiting_times_rho_D = np.zeros((3, steps,n_samples))
#%%
for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
        env = simpy.Environment()
        servers3 = simpy.PriorityResource(env, capacity=n_servers[i])
        waiting_times = []
        setup3 = DES_MD_LT(env, arrival_rate[i], service_rate[j], servers3, waiting_times, n_samples,'deterministic')
        env.run(until=setup3.num_samples_count)
        waiting_times_rho_D[i, j, :] = setup3.waiting_times[:n_samples]

mean_waiting_times_D = np.zeros((len(n_servers), steps))

for i in range(len(n_servers)):
    for j in range(steps):
        mean_waiting_times_D[i, j] = np.mean(waiting_times_rho_D[i, j, :])

np.savetxt("mean_waiting_time_0.5-0.95_MDN.csv", mean_waiting_times_D, delimiter=",")

# %%

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

mean_waiting_times_D = genfromtxt("mean_waiting_time_0.5-0.95_MDN.csv", delimiter=',')


for i in range(len(n_servers)):
    plt.scatter(p_range, mean_waiting_times_D[i], marker='o',label = '%s server(s) computatioanl value' % n_servers[i], linewidth = 2)
colors = ['cornflowerblue', 'forestgreen', 'lightcoral']
for i in range(3):
    plt.plot(rho_range, theoretical_det[i, :], linestyle = '--', color = colors[i] ,alpha = 0.9, label = 'theoretical value for {} server(s)'.format(i + 1))
plt.xlabel(r'$\rho$', fontsize = 17)
plt.ylabel("Waiting time", fontsize = 14)
plt.title("Waiting times for M/D/n vs amount of servers available", fontsize  = 15)
plt.legend(fontsize = 13)
    
# %%
for i in range(len(n_servers)):
    plt.plot(service_rate, mean_waiting_times[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()
    
for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
        env = simpy.Environment()
        servers1 = simpy.PriorityResource(env, capacity=n_servers[i])
        waiting_times = []
        setup1 = Setup_shortestjob(env, arrival_rate[i], service_rate[j], servers1, waiting_times, n_samples,random.expovariate)
        env.run(until=setup1.n_samples_reached)
        waiting_times_rho_s[i, j, :] = setup1.waiting_times[:n_samples]

#%%
mean_waiting_times = np.zeros((len(n_servers), steps))

for i in range(len(n_servers)):
    for j in range(steps):
        mean_waiting_times[i, j] = np.mean(waiting_times_rho_s[i, j, :])


mean_waiting_times.shape
# %%
for i in range(len(n_servers)):
    plt.scatter(p_range, mean_waiting_times[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()

#%%
rho_range = np.linspace(p_min, p_max, 1000)
mu_range = 1 / rho_range
theory = []

for rho, mu in zip(rho_range, mu_range):
    theory.append(Functions.SPTF(rho, mu, 1))

#%%
for i in range(len(n_servers)):
    plt.plot(p_range, mean_waiting_times[i], label = '%s servers' % n_servers[i])
plt.plot(rho_range , theory, linestyle = '--', linewidth = 3)
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()
#%%
#%#%#%#%%#%#%#%#%#%#%#%#%#%#%#%%#%#%# new new new new new new new new #%#%#%#%#%#%%#

n_samples = 100000
n_servers = np.array([1, 2, 4])
steps = 100
arrival_rate = n_servers
p_min = 0.5
p_max = 0.95
p_range = np.linspace(p_min, p_max, steps)
service_rate = (1 / p_range)

waiting_times_rho_D = np.zeros((3, steps,n_samples))

#%%
for i in range(len(n_servers)):
    for j in tqdm(range(steps), desc=f'calculate waiting times for n_server {n_servers[i]}'):
        env = simpy.Environment()
        servers1 = simpy.Resource(env, capacity=n_servers[i])
        waiting_times = []
        setup1 = Setup_samples(env, arrival_rate[i], processing_capacity[j], servers1, waiting_times, n_samples,deterministic)
        env.run(until=setup1.n_samples_reached)
        waiting_times_rho_D[i, j, :] = setup1.waiting_times[:n_samples]
# np.save(f'results/waiting_times_rho_md', waiting_times_rho_md)

#%%

mean_waiting_times_D = np.zeros((len(n_servers), steps))

for i in range(len(n_servers)):
    for j in range(steps):
        mean_waiting_times_D[i, j] = np.mean(waiting_times_rho_D[i, j, :])


for i in range(len(n_servers)):
    plt.plot(p_range, mean_waiting_times_D[i], label = '%s servers' % n_servers[i])
plt.xlabel("p")
plt.ylabel("Waiting time")
plt.title("Waiting times vs amount of servers available")
plt.legend()
#%%