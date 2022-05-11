#!/usr/bin/env python
# coding: utf-8

# In[1]:
import Generate_Data as gen
import numpy as np
import emcee
import ast
from multiprocessing import Pool
import math
from scipy.integrate import odeint

def log_likelihood_ic(theta, model, times, y):
    # theta = (a, sigma, ic)
    if len(theta) == 3:
        cal_a, cal_sigma, cal_ic = theta
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a]]))
    # theta = (a, b, sigma, ic)
    elif len(theta) == 4:
        cal_a, cal_b, cal_sigma, cal_ic = theta
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a, cal_b]]))
    # theta = (a, b, c, sigma, ic)
    elif len(theta) == 5:
        cal_a, cal_b, cal_c, cal_sigma, cal_ic = theta
        y_model = odeint(model, t=times, y0=cal_ic, args=tuple([[cal_a, cal_b, cal_c]]))
    # theta = (a, b, c, d, sigma, ic)
    elif len(theta) == 6:
        cal_a, cal_b, cal_c, cal_d, cal_sigma, cal_ic = theta
        y_model = odeint(model, t=times, y0=cal_ic,
                         args=tuple([[cal_a, cal_b, cal_c, cal_d]]))
    # theta = (a, b, c, d, e, sigma, ic)
    elif len(theta) == 7:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_sigma, cal_ic = theta
        y_model = odeint(model, t=times, y0=cal_ic,
                         args=tuple(
                             [[cal_a, cal_b, cal_c, cal_d, cal_e]]))
    # theta = (a, b, c, d, e, f, sigma, ic)
    elif len(theta) == 8:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_f, cal_sigma, cal_ic = theta
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
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_sigma < bounds[1][1]                 and bounds[2][0] < cal_ic < bounds[2][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, sigma, ic)
    elif len(theta) == 4 and len(bounds) == 4:
        cal_a, cal_b, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b < bounds[1][1]                 and bounds[2][0] < cal_sigma < bounds[2][1] and bounds[3][0] < cal_ic <                 bounds[3][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, sigma, ic)
    elif len(theta) == 5 and len(bounds) == 5:
        cal_a, cal_b, cal_c, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b < bounds[1][1]                 and bounds[2][0] < cal_c < bounds[2][1] and bounds[3][0] < cal_sigma <                 bounds[3][1] and bounds[4][0] < cal_ic < bounds[4][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, d, sigma, ic)
    elif len(theta) == 6 and len(bounds) == 6:
        cal_a, cal_b, cal_c, cal_d, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b < bounds[1][1] and bounds[2][0] < cal_c < bounds[2][1] and bounds[3][0] < cal_d <                 bounds[3][1] and bounds[4][0] < cal_sigma <                 bounds[4][1] and bounds[5][0] < cal_ic < bounds[5][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, d, e, sigma, ic)
    elif len(theta) == 7 and len(bounds) == 7:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b <                 bounds[1][1] and bounds[2][0] < cal_c < bounds[2][1] and bounds[3][0] < cal_d < bounds[3][1] and                 bounds[4][0] < cal_e <                 bounds[4][1] and bounds[5][0] < cal_sigma < bounds[5][1] and                 bounds[6][0] < cal_ic < bounds[6][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, d, e, f, sigma, ic)
    elif len(theta) == 8 and len(bounds) == 8:
        cal_a, cal_b, cal_c, cal_d, cal_e, cal_f, cal_sigma, cal_ic = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b <                 bounds[1][1] and bounds[2][0] < cal_c < bounds[2][1] and bounds[3][0] < cal_d < bounds[3][1] and                 bounds[4][0] < cal_e <                 bounds[4][1] and bounds[5][0] < cal_f < bounds[5][1] and                 bounds[6][0] < cal_sigma < bounds[6][1] and bounds[7][0] < cal_ic < bounds[7][1]:
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
    np.random.seed(20394)
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


# Calibrate model
def calibrate(nchains, model, init_bounds, num_steps, discard, percentile,
              obs_data, times, chain_name):
    pos = initialize_chains(nchains, init_bounds)
    sampler = run_MCMC(pos, num_steps, model, init_bounds, obs_data, times)
    flat_samples = flatten(sampler, discard)
    final_parameters = percentile_results(flat_samples, percentile)
    # save final parameters for 16, 50, 84 percentiles
    final_parameters16 = percentile_results(flat_samples, 16)
    final_parameters50 = percentile_results(flat_samples, 50)
    final_parameters84 = percentile_results(flat_samples, 84)
    #np.savetxt('16_' + str(chain_name) + '.txt', final_parameters16)
    np.savetxt('50_' + str(chain_name) + '.txt', final_parameters50)
    #np.savetxt('84_' + str(chain_name) + '.txt', final_parameters84)
  
    return pos, sampler, flat_samples, final_parameters


# In[2]:


# Returns final model given model name, test number, perturbation
def final_param(model, test, noise, percentile):
    fname = str(percentile) + "_" + model + "_test" + str(test) + "_" + str(noise) + ".txt"
    params = []
    with open(fname) as final_params:
        for line in final_params:
            val = float(line)
            params.append(val)
    print(params)
    return params

# Calculating Bayesian information criterion
def bic(theta, model, times, obs_data):
    k = len(theta)
    n = len(obs_data)
    return k * np.log(n) - 2 * log_likelihood_ic(theta, model, times, obs_data)

# Truncate floats
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

# Returns BIC of all 6 models
def all_BIC(test_num, test_noise, percentile=50):
    test_name = "test" + str(test_num) + "_" + str(test_noise)
    obs_fname = "obsdata_" + test_name + ".txt"
    all_data = np.loadtxt(obs_fname)
    times = all_data.T[0]
    obs_data = all_data.T[1]
    obs_data = obs_data.reshape((-1,1))
    all_bic00 = [0]*6
    
    M3_param = final_param("M3", test_num, test_noise, percentile)
    M5_param = final_param("M5", test_num, test_noise, percentile)
    M7_param = final_param("M7", test_num, test_noise, percentile)
    M8_param = final_param("M8", test_num, test_noise, percentile)
    M9_param = final_param("M9", test_num, test_noise, percentile)
    M10_param = final_param("M10", test_num, test_noise, percentile)
   
    all_bic00[0] = bic(M3_param, gen.M3, times, obs_data)
    all_bic00[1] = bic(M5_param, gen.M5, times, obs_data)
    all_bic00[2] = bic(M7_param, gen.M7, times, obs_data)
    all_bic00[3] = bic(M8_param, gen.M8, times, obs_data)
    all_bic00[4] = bic(M9_param, gen.M9, times, obs_data)
    all_bic00[5] = bic(M10_param, gen.M10, times, obs_data)
    
    for i in range(len(all_bic00)):
        all_bic00[i] = truncate(all_bic00[i],3)
    return all_bic00

    


# In[3]:


# All model bounds
M3_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.0, 1000.0], [700, 3000]]
M5_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]
M7_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.0, 1000.0], [700, 3000]]
M8_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.001, 0.008], [0.0, 1000.0], [700, 3000]]
M9_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]
M10_bounds = [[0.001, 0.065], [0.0, 0.03], [15000, 90000.0], [0.008333, 0.1], [0.0, 1000.0], [700, 3000]]



#-------------------- Read the data --------------------
np.random.seed(20394)

for test_num in range(30):
    test_name = "test" + str(test_num+1)
    # Generate initial parameters
    times = np.arange(0, 500, 10)
    true_p = np.random.uniform(0.001, 0.065)
    true_d = np.random.uniform(0.0, 0.03)
    true_K = np.random.uniform(15000, 90000)
    true_r = np.random.uniform(0.008333, 0.1)
    true_g = np.random.uniform(0.001, 0.008)
    true_ic = np.random.uniform(700, 3000)


    true_param = [true_p, true_d, true_K, true_ic]
    # Save true parameters
    fname1 = "true_param_" + test_name
    file = open(fname1, "w") 
    file.write(str(true_param) + "\n")
    file.close()

    # Generate observed data
    perturb_perc = [2.5, 5.0, 7.5, 10.0]
    for i in range (len(perturb_perc)):
        sigma = perturb_perc[i] / 100
        true_data = gen.gen_true_data(gen.test_M7, [true_ic], times, [true_p, true_d, true_K])
        obs_data = gen.gen_obs_data(true_data, sigma)
        obs_data[obs_data < 0] = 0

        # Save observed data
        chain_name = "_" + test_name + "_" + str(perturb_perc[i]) 
        fname = "obsdata" + chain_name + ".txt"
        file = open(fname, "w")
        combine = np.zeros([len(times),2])
        for i in range(len(times)):
            combine[i] = [times[i], obs_data[i][0]]
            file.write(str(combine[i][0]) + " " + str(combine[i][1]) + "\n")
        file.close()
        
        # cut obs_data off at 300 hours
        obs_data = obs_data[0:30]
        times = np.arange(0, 300, 10)
    
        
        # Name model chains
        M3_name = "M3" + chain_name
        M5_name = "M5" + chain_name
        M7_name = "M7" + chain_name
        M8_name = "M8" + chain_name
        M9_name = "M9" + chain_name
        M10_name = "M10" + chain_name
        
        # Run all models in parallel
        with Pool() as pool:
             pool.starmap(calibrate,[(15, gen.M3, M3_bounds, 10000, 3000, 50, obs_data, times, M3_name),
                                     (15, gen.M5, M5_bounds, 10000, 3000, 50, obs_data, times, M5_name),
                                     (15, gen.M7, M7_bounds, 10000, 3000, 50, obs_data, times, M7_name),
                                     (15, gen.M8, M8_bounds, 10000, 3000, 50, obs_data, times, M8_name),
                                     (15, gen.M9, M9_bounds, 10000, 3000, 50, obs_data, times, M9_name),
                                     (15, gen.M10, M10_bounds, 10000, 3000, 50, obs_data, times, M10_name)])
       


