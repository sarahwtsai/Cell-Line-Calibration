#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import glob
import Generate_Data as gen
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

import math
from scipy.integrate import odeint
from tabulate import tabulate
from scipy.optimize import minimize
from IPython.display import display, Math
from scipy.optimize import Bounds


def log_likelihood(theta, model, times, y):
    # theta = (a, sigma)
    if len(theta) == 2:
        cal_a, cal_sigma = theta
        y_model = odeint(model, t=times, y0=true_ic, args=tuple([[cal_a]]))
    # theta = (a, b, sigma)
    elif len(theta) == 3:
        cal_a, cal_b, cal_sigma = theta
        y_model = odeint(model, t=times, y0=true_ic, args=tuple([[cal_a, cal_b]]))
    # theta = (a, b, c, sigma)
    elif len(theta) == 4:
        cal_a, cal_b, cal_c, cal_sigma = theta
        y_model = odeint(model, t=times, y0=true_ic,
                         args=tuple([[cal_a, cal_b, cal_c]]))
    else:
        print("Log likelihood: Choose appropriate number of parameters")
        return -1

    variance = cal_sigma * cal_sigma
    return -0.5 * np.sum((y - y_model) ** 2 / variance + np.log(2 * np.pi) +
                         np.log(variance))


def n_log_likelihood(theta, model, times, y):
    return -log_likelihood(theta, model, times, y)


def log_prior(theta, bounds):
    # theta = (a, sigma)
    if len(theta) == 2 and len(bounds) == 2:
        cal_a, cal_sigma = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_sigma <                 bounds[1][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, sigma)
    elif len(theta) == 3 and len(bounds) == 3:
        cal_a, cal_b, cal_sigma = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b <                 bounds[1][1]                 and bounds[2][0] < cal_sigma < bounds[2][1]:
            return 0.0
        return -np.inf
    # theta = (a, b, c, sigma)
    elif len(theta) == 4 and len(bounds) == 4:
        cal_a, cal_b, cal_c, cal_sigma = theta
        if bounds[0][0] < cal_a < bounds[0][1] and bounds[1][0] < cal_b <                 bounds[1][1]                 and bounds[2][0] < cal_c < bounds[2][1] and bounds[3][
            0] < cal_sigma < \
                bounds[3][1]:
            return 0.0
        return -np.inf
    else:
        print("Log prior: Choose appropriate number of parameters and bounds")
        return -1


def log_probability(theta, model, times, y, bounds):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, times, y)


# Determine number of chains and generate initial parameter values
def initialize_chains(num_chains, bounds):
    chains = num_chains
    ndim = len(bounds)
    chain_init = np.zeros((chains, ndim))
    for i in range(chains):
        for j in range(ndim):
            chain_init[i][j] = np.random.uniform(bounds[j][0], bounds[j][1])
    return chain_init


# Run MCMC
def run_MCMC(pos, num_steps, model, bounds):
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(model,
                                          times, obs_data, bounds))
    sampler.run_mcmc(pos, num_steps, progress=True)
    return sampler


# Calculating Bayesian information criterion
def bic(theta, model, times, obs_data):
    k = len(theta)
    n = len(obs_data)
    return k * np.log(n) - 2 * log_likelihood(theta, model, times, obs_data)

# Calculating average error between two curves
def avg_perc_error(avg_cal_y, obs_data):
    avg_err_percent = np.sum(np.abs(obs_data - avg_cal_y) / obs_data) / len(
        obs_data) * 100
    return avg_err_percent

# Calculating concordance correlation coefficient
def ccc(model, obs_data):
    sxy = np.sum((model - model.mean()) * (obs_data - obs_data.mean())) / model.shape[0]
    rhoc = 2 * sxy / (np.var(model) + np.var(obs_data) + (model.mean() - obs_data.mean()) ** 2)
    return rhoc

# Display trace plots
def display_traceplot(pos, sampler, labels=None):
    nwalkers, ndim = pos.shape
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    if labels == None:
        if ndim == 2:
            labels = ["a", "Sigma"]
        elif ndim == 3:
            labels = ["a", "b", "Sigma"]
        elif ndim == 4:
            labels = ["a", "b", "c", "Sigma"]
        else:
            print("Trace Plot: Enter appropriate number of parameters")
            return -1

    if ndim != len(labels):
        print("Trace plot: Number of labels and parameters do not match")
        return -1

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step Number")
   
   
# Display histogram and contour plots
def display_histogram(flat_samples, labels):
    fig = corner.corner(flat_samples, labels=labels,
                        truths=[true_a, true_b, true_sigma, true_ic])
                        
                        
# Return parameter results (percentile)
def percentile_results(flat_samples, percentile):
    result = np.zeros(len(flat_samples[0]))
    for i in range(len(flat_samples[0])):
        result[i] = np.percentile(flat_samples[:, i], percentile)
    return result


# Autocorrelation time
def autocorr_time(sampler):
    tau = sampler.get_autocorr_time()
    return tau


# Flatten chains
def flatten(sampler, discard_num):
    flat_samples = sampler.get_chain(discard=discard_num, flat=True)
    return flat_samples


# Calculating percent error between parameter results and original parameters
def param_perc_error(original, results):
    error = (np.abs(original - results)) / original * 100
    return error


# Update prior bounds
def update_bounds(init_bounds, final_parameters):
    new_bounds = np.copy(init_bounds)
    for i in range(len(final_parameters)):
        if (np.abs(init_bounds[i][1] - final_parameters[i]) / init_bounds[i][
            1]) < 0.15:
            new_bounds[i][1] = 5 * init_bounds[i][1]
    return new_bounds


# Calibrate model
def calibrate(nchains, model, init_bounds, num_steps, discard, percentile, chain_name):
    pos = initialize_chains(nchains, init_bounds)
    sampler = run_MCMC(pos, num_steps, model, init_bounds)
    flat_samples = flatten(sampler, discard)
    final_parameters = percentile_results(flat_samples, percentile)
    new_bounds = update_bounds(init_bounds, final_parameters)

    if not np.allclose(new_bounds, init_bounds):
        return calibrate(nchains, model, new_bounds, num_steps, discard,
                         percentile, chain_name)
    else:
        # save chains
        samples = sampler.get_chain()
        np.save('BT474_control_chains_' + str(chain_name) + '.npy', samples)
        # save flat chains
        flat_samples = flatten(sampler, discard)
        np.save('BT474_control_fchains_' + str(chain_name) + '.npy', flat_samples)
        return pos, sampler, flat_samples, final_parameters


# In[2]:


# Find means of all scenarios
plt.rcParams['font.size'] = '8'
#np.random.seed(203945)

FilenamesList = sorted(glob.glob('./cell_BT474_*.txt'))
data_list = []
for file in FilenamesList:
    u = np.loadtxt(file)
    data_list.append(u)
treatment = 45.416666667

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


# In[3]:


# Calibrate data without treatment
control = means[0]
times, obs_data1 = control[0], control[1]
obs_data = []
for data in obs_data1:
    obs_data.append([data])
obs_data = np.array(obs_data)

true_ic = obs_data[0]

# All model bounds
exp_bounds = [[0.0, 0.1], [0.0, 1000.0]]
men_bounds = [[0.0, 0.1], [0.0, 10.0], [0.0, 1000.0]]
log_bounds = [[0.0, 0.1], [0.0, 10000.0], [0.0, 1000.0]]
lin_bounds = [[0.0, 1.0], [0.0, 1000.0], [0.0, 100.0]]
surf_bounds = [[0.0, 1.0], [0.0, 1000.0], [0.0, 100.0]]
gomp_bounds = [[0.0, 1.0], [0.0, 10000.0], [0.0, 100.0]]
bert_bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 100.0]]


# In[4]:


# Run MCMC and calculate BIC for model selection
# Exponential
exp_cal = calibrate(10, gen.exponential, exp_bounds, 12000, 450, 50, "EXP")
exp_param = exp_cal[3]
exp_final_model = gen.test_exponential([true_ic], times, [exp_param[0]])
exp_bic = bic(exp_param, gen.exponential, times, obs_data)


# Mendelsohn
men_cal = calibrate(10, gen.mendelsohn, men_bounds, 12000, 450, 50, "MEN")
men_param = men_cal[3]
men_final_model = gen.test_mendelsohn([true_ic], times,
                                      [men_param[0], men_param[1]])
men_bic = bic(men_param, gen.mendelsohn, times, obs_data)


# Logistic
log_cal = calibrate(10, gen.logistic, log_bounds, 12000, 450, 50, "LOG")
log_param = log_cal[3]
log_final_model = gen.test_logistic([true_ic], times,
                                    [log_param[0], log_param[1]])
log_bic = bic(log_param, gen.logistic, times, obs_data)


# Linear
lin_cal = calibrate(10, gen.linear, lin_bounds, 12000, 450, 50, "LIN")
lin_param = lin_cal[3]
lin_final_model = gen.test_linear([true_ic], times,
                                  [lin_param[0], lin_param[1]])
lin_bic = bic(lin_param, gen.linear, times, obs_data)


# Surface
surf_cal = calibrate(10, gen.surface, surf_bounds, 12000, 450, 50, "SURF")
surf_param = surf_cal[3]
surf_final_model = gen.test_surface([true_ic], times,
                                    [surf_param[0], surf_param[1]])
surf_bic = bic(surf_param, gen.surface, times, obs_data)


# Gompertz
gomp_cal = calibrate(10, gen.gompertz, gomp_bounds, 12000, 450, 50, "GOMP")
gomp_param = gomp_cal[3]
gomp_final_model = gen.test_gompertz([true_ic], times,
                                     [gomp_param[0], gomp_param[1]])
gomp_bic = bic(gomp_param, gen.gompertz, times, obs_data)


# Bertalanffy
bert_cal = calibrate(10, gen.bertalanffy, bert_bounds, 12000, 450, 50, "BERT")
bert_param = bert_cal[3]
bert_final_model = gen.test_bertalanffy([true_ic], times,
                                        [bert_param[0], bert_param[1]])
bert_bic = bic(bert_param, gen.bertalanffy, times, obs_data)


# Display final results
fig, ax = plt.subplots(dpi=120)
# plt.ylim(0, 5000)
plt.title("BT474 Control Calibration")
plt.scatter(times, obs_data, zorder=100, label='Measured data', color='red')
plt.plot(times, exp_final_model, label='Exponential', alpha=0.5)
plt.plot(times, men_final_model, label='Mendelsohn', alpha=0.5)
plt.plot(times, log_final_model, label='Logistic', alpha=0.5)
plt.plot(times, lin_final_model, label='Linear', alpha=0.5)
plt.plot(times, surf_final_model, label='Surface', alpha=0.5)
plt.plot(times, gomp_final_model, label='Gompertz', alpha=0.5)
plt.plot(times, bert_final_model, label='Bertalanffy', alpha=0.5)
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Cell number')
# plt.savefig("bc_fit.pdf")
plt.show()




