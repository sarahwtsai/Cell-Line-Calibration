#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Generate_Data as gen
import numpy as np
import emcee
import sys
from scipy.integrate import odeint
from IPython.display import display, Math
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
import os
import glob

import math
from scipy.integrate import odeint
from scipy.optimize import minimize
from IPython.display import display, Math
from scipy.optimize import Bounds


def log_likelihood_ic(theta, model, times, y):
    # theta = (a, sigma, ic)
    if len(theta) == 3:
        cal_a, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a]]))
    # theta = (a, b, sigma, ic)
    elif len(theta) == 4:
        cal_a, cal_b, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a, cal_b]]))
    # theta = (a, b, c, sigma, ic)
    elif len(theta) == 5:
        cal_a, cal_b, cal_c, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a, cal_b, cal_c]]))
    # theta = (a, b, c, d, sigma, ic)
    elif len(theta) == 6:
        cal_a, cal_b, cal_c, cal_d, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic,
                         args=tuple([[cal_a, cal_b, cal_c, cal_d]]))
    # theta = (a, b, c, d, e, sigma, ic)
    elif len(theta) == 7:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic,
                         args=tuple(
                             [[cal_a, cal_b, cal_c, cal_d, cal_e]]))
    # theta = (a, b, c, d, e, f, sigma, ic)
    elif len(theta) == 8:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_f, cal_sigma, cal_ic = theta
        theta[0] = 1.10836435e-02
        theta[2] = 1.29165896e+05
        y_model = odeint(model, t=times, y0=cal_ic,
                         args=tuple(
                             [[cal_a, cal_b, cal_c, cal_d, cal_e, cal_f]]))
    else:
        print("Log likelihood: Choose appropriate number of parameters")
        return -1

    variance = cal_sigma * cal_sigma
    return -0.5 * np.sum((y - y_model) ** 2 / variance + np.log(2 * np.pi) +
                         np.log(variance))


def n_log_likelihood_ic(theta, model, times, y):
    return -log_likelihood_ic(theta, model, times, y)


def log_prior(theta, bounds):
    # theta = (a, sigma, ic)
    if len(theta) == 3 and len(bounds) == 3:
        cal_a, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_sigma <                 bounds[1][1]                 and bounds[2][0] < cal_ic < bounds[2][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, sigma, ic)
    elif len(theta) == 4 and len(bounds) == 4:
        cal_a, cal_b, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b <                 bounds[1][1]                 and bounds[2][0] < cal_sigma < bounds[2][1] and bounds[3][
            0] < cal_ic < \
                bounds[3][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, sigma, ic)
    elif len(theta) == 5 and len(bounds) == 5:
        cal_a, cal_b, cal_c, cal_sigma, cal_ic = theta
        if bounds[1][0] < cal_b < bounds[1][1] and bounds[3][0] < cal_sigma <                 bounds[3][1] and bounds[4][0] < cal_ic < bounds[4][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, d, sigma, ic)
    elif len(theta) == 6 and len(bounds) == 6:
        cal_a, cal_b, cal_c, cal_d, cal_sigma, cal_ic = theta
        if bounds[1][0] < cal_b < bounds[1][1] and bounds[3][0] < cal_d <                 bounds[3][1] and bounds[4][0] < cal_sigma <                 bounds[4][1] and bounds[5][0] < cal_ic < bounds[5][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, d, e, sigma, ic)
    elif len(theta) == 7 and len(bounds) == 7:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_sigma, cal_ic = theta
        if bounds[1][0] < cal_b <                 bounds[1][1] and bounds[3][0] < cal_d < bounds[3][1] and                 bounds[4][0] < cal_e <                 bounds[4][1] and bounds[5][0] < cal_sigma < bounds[5][1] and                 bounds[6][0] < cal_ic < bounds[6][1]:
            return 0.0
        return -np.inf
    elif len(theta) == 8 and len(bounds) == 8:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_f, cal_sigma, cal_ic = theta
        if bounds[1][0] < cal_b <                 bounds[1][1] and bounds[3][0] < cal_d < bounds[3][1] and                 bounds[4][0] < cal_e <                 bounds[4][1] and bounds[5][0] < cal_f < bounds[5][1] and                 bounds[6][0] < cal_sigma < bounds[6][1] and bounds[7][0] < cal_ic < bounds[7][1]:
            return 0.0
        return -np.inf
    else:
        print("Log prior: Choose appropriate number of parameters and bounds")
        return -1


def log_probability(theta, model, times, y, bounds):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ic(theta, model, times, y)


# Determine number of chains and generate initial parameter values
def initialize_chains(num_chains, bounds):
    chains = num_chains
    ndim = len(bounds)
    chain_init = np.zeros((chains, ndim))
    for i in range(chains):
        for j in range(ndim):
            chain_init[i][j] = np.random.uniform(bounds[j][0], bounds[j][1])
    return chain_init


# Run MCMC (serial)
def run_MCMC(pos, num_steps, model, bounds, obs_data, times):
    nwalkers, ndim = pos.shape
    #np.random.seed(20394)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,args=(model, times, obs_data, bounds))
    sampler.run_mcmc(pos, num_steps, progress=True)
    return sampler

# Autocorrelation time
def autocorr_time(sampler):
    tau = sampler.get_autocorr_time()
    return tau


# Flatten chains
def flatten(sampler, discard_num):
    flat_samples = sampler.get_chain(discard=discard_num, flat=True)
    return flat_samples


# Return parameter results (percentile)
def percentile_results(flat_samples, percentile):
    result = np.zeros(len(flat_samples[0]))
    for i in range(len(flat_samples[0])):
        result[i] = np.percentile(flat_samples[:, i], percentile)
    return result


# Calibrate model, NO update bounds
def calibrate(nchains, model, init_bounds, num_steps, discard, percentile,
              obs_data, times, chain_name):
    pos = initialize_chains(nchains, init_bounds)
    sampler = run_MCMC(pos, num_steps, model, init_bounds, obs_data, times)
    # save chains
    samples = sampler.get_chain()
    np.save('chains_' + str(chain_name) + '.npy', samples)
    # save flat chains
    flat_samples = flatten(sampler, discard)
    np.save('flat_chains_' + str(chain_name) + '.npy', flat_samples)
    final_parameters = percentile_results(flat_samples, percentile)
    return pos, sampler, flat_samples, final_parameters

    

# In[2]:


# All model bounds
M3_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.0, 1000.0], [700, 3000]]
M5_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]
M7_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.0, 1000.0], [700, 3000]]
M8_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.001, 0.008], [0.0, 1000.0], [700, 3000]]
M9_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]
M10_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]

M3_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]
M5_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]
M7_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]
M8_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.001, 0.008], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]
M9_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]
M10_new_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.001, 0.065], [0.0, 1000.0], [700, 3000]]


# In[3]:


# Find means of all scenarios
FilenamesList = sorted(glob.glob('./cell_BT474_*.txt'))
data_list = []
for file in FilenamesList:
    u = np.loadtxt(file)
    data_list.append(u)
treatment = 53.58333333

means = []
for i in range(len(data_list)):
    mean_data = np.mean(np.delete(data_list[i],0,1),axis=1)
    nm_data = np.stack([data_list[i].T[0], mean_data])
    means.append(nm_data)

# Crop the data above 600 hours
for j in range(len(means)):
    pos = np.argmax(means[j][0]>600)
    if pos>0:
        means[j] = means[j][:,0:pos]
# Crop the data above 20000 cells
for j in range(len(means)):
    pos = np.argmax(means[j][1]>20000)
    if pos>0:
        means[j] = means[j][:,0:pos]


# In[4]:


conc_levels = ["000", "010", "015", "020", "028", "035", "043", "050", "060", "120" ]
for conc in range(len(conc_levels)):
    # np.random.seed(20394)
    chain_name = conc_levels[conc] + "_"
    all_data = means[conc]
    times, obs_data1 = all_data[0], all_data[1]
    
    # Convert to np.array
    obs_data = []
    for data in obs_data1:
        obs_data.append([data])
    obs_data = np.array(obs_data)
    
    
    # Name model chains
    M3_name = chain_name + "M3" 
    M5_name = chain_name + "M5"
    M7_name = chain_name + "M7" 
    M8_name = chain_name + "M8" 
    M9_name = chain_name + "M9" 
    M10_name = chain_name + "M10"
    # Name new model chains
    nM3_name = chain_name + "newM3" 
    nM5_name = chain_name + "newM5"
    nM7_name = chain_name + "newM7" 
    nM8_name = chain_name + "newM8" 
    nM9_name = chain_name + "newM9" 
    nM10_name = chain_name + "newM10" 
    
        
    # Run 12 models in parallel
    with Pool() as pool:
        pool.starmap(calibrate,[(15, gen.M3, M3_bounds, 10000, 3000, 50, obs_data, times, M3_name),
                                (15, gen.M5, M5_bounds, 10000, 3000, 50, obs_data, times, M5_name),
                                (15, gen.M7, M7_bounds, 10000, 3000, 50, obs_data, times, M7_name),
                                (15, gen.M8, M8_bounds, 10000, 3000, 50, obs_data, times, M8_name),
                                (15, gen.M9, M9_bounds, 10000, 3000, 50, obs_data, times, M9_name),
                                (15, gen.M10, M10_bounds, 10000, 3000, 50, obs_data, times, M10_name),
                                (15, gen.M3_new, M3_new_bounds, 10000, 3000, 50, obs_data, times, nM3_name),
                                (15, gen.M5_new, M5_new_bounds, 10000, 3000, 50, obs_data, times, nM5_name),
                                (15, gen.M7_new, M7_new_bounds, 10000, 3000, 50, obs_data, times, nM7_name),
                                (15, gen.M8_new, M8_new_bounds, 10000, 3000, 50, obs_data, times, nM8_name),
                                (15, gen.M9_new, M9_new_bounds, 10000, 3000, 50, obs_data, times, nM9_name),
                                (15, gen.M10_new, M10_new_bounds, 10000, 3000, 50, obs_data, times, nM10_name)])

