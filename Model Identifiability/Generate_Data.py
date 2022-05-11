import numpy as np
import math
from scipy.integrate import odeint

# Time delay on cell death
T = 45.416666667


# theta = (p, d, K)
# y = (N)
# dN/dt = (p-d)N(1-N/K)
def M3(y, t, theta):
    if t < T:
        tumor = (theta[0] - 0) * y[0] * (1 - y[0] / theta[2])
    else:
        tumor = (theta[0] - theta[1]) * y[0] * (1 - y[0] / theta[2])
    return tumor


def test_M3(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    y_M3 = odeint(M3, cells_ic, t, args=tuple([[p, d, K]]))
    return y_M3


# theta = (p, d, K, r)
# y = (N)
# dN/dt = (p-d*t*e^(-rt))N(1-N/K)
def M5(y, t, theta):
    if t < T:
        tumor = (theta[0]) * y[0] * (1 - y[0] / theta[2])
    else:
        tumor = (theta[0] - theta[1] * (t-T) * np.exp(-theta[3] * (t-T))) * y[
            0] * (1 - y[0] / theta[2])
    return tumor


def test_M5(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    y_M5 = odeint(M5, cells_ic, t, args=tuple([[p, d, K, r]]))
    return y_M5


# theta = (p, d, K)
# y = (N)
# dN/dt = pN(1-N/K)-dN
def M7(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * y[0]
    else:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0]
    return tumor


def test_M7(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    y_M7 = odeint(M7, cells_ic, t, args=tuple([[p, d, K]]))
    return y_M7


# theta = (p, d, K, g)
# y = (N)
# dN/dt = pN(1-N/K)-dN/(1+gt)
def M8(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                y[0] / (1 + theta[3] * (t-T))
    else:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0] / (
                    1 + theta[3] * (t-T))
    return tumor


def test_M8(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    g = theta[3]
    y_M8 = odeint(M8, cells_ic, t, args=tuple([[p, d, K, g]]))
    return y_M8


# theta = (p, d, K, r)
# y = (N)
# dN/dt = pN(1-N/K)-de^(-rt)N
def M9(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                y[0] * np.exp(-theta[3] * (t-T))
    else:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0] * np.exp(
            -theta[3] * (t-T))
    return tumor


def test_M9(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    y_M9 = odeint(M9, cells_ic, t, args=tuple([[p, d, K, r]]))
    return y_M9



# theta = (p, d, K, r)
# y = (N)
# dN/dt = pN(1-N/K)-dte^(-rt)N
def M10(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                theta[3] * y[0] * np.exp(1 - theta[3] * (t-T))
    else:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - theta[1] *(t-T)* y[0] * np.exp(-theta[3] * (t-T))
    return tumor


def test_M10(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    y_M10 = odeint(M10, cells_ic, t, args=tuple([[p, d, K, r]]))
    return y_M10


def gen_true_data(model, y, t, p):
    return model(y, t, p)

def gen_obs_data(y, sigma, seed_num = 20394):
    np.random.seed(seed_num)
    obs_data = []
    for cells in y:
        cells = cells*(1-np.random.uniform(-sigma,sigma))
        obs_data.append(cells)
    obs_data = np.asarray(obs_data)
    
    np.random.seed()
    return obs_data

