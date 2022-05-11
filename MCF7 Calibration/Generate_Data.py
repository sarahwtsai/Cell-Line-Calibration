import numpy as np
import math
from scipy.integrate import odeint

T = 45.416666667

# p = (r)
# y = (N)
# dN/dt = rN
def exponential(y, t, p):
    tumor = p[0] * y[0]
    return tumor


def test_exponential(y, t, p):
    cells_ic = y[0]
    r = p[0]
    y_exp = odeint(exponential, cells_ic, t, args=tuple([[r]]))
    return y_exp


# p = (r, b)
# y = (N)
# dN/dt = rN^b
def mendelsohn(y, t, p):
    tumor = p[0] * y[0] ** p[1]
    return tumor


def test_mendelsohn(y, t, p):
    cells_ic = y[0]
    r = p[0]
    b = p[1]
    y_men = odeint(mendelsohn, cells_ic, t, args=tuple([[r, b]]))
    return y_men


# p = (r,K)
# y = (N)
# dN/dt = rN*(1-N/K)
def logistic(y, t, p):
    tumor = p[0] * y[0] * (1.0 - y[0] / p[1])
    return tumor


def test_logistic(y, t, p):
    cells_ic = y[0]
    r = p[0]
    K = p[1]
    y_log = odeint(logistic, cells_ic, t, args=tuple([[r, K]]))
    return y_log


# p = (r, b)
# y = (N)
# dN/dt = rN/(N+b)
def linear(y, t, p):
    tumor = p[0] * y[0] / (y[0] + p[1])
    return tumor


def test_linear(y, t, p):
    cells_ic = y[0]
    r = p[0]
    b = p[1]
    y_lin = odeint(linear, cells_ic, t, args=tuple([[r, b]]))
    return y_lin


# p = (r, b)
# y = (N)
# dN/dt = rN/(N+b)^(1/3)
def surface(y, t, p):
    tumor = p[0] * y[0] / ((y[0] + p[1]) ** (1 / 3))
    return tumor


def test_surface(y, t, p):
    cells_ic = y[0]
    r = p[0]
    b = p[1]
    y_surf = odeint(surface, cells_ic, t, args=tuple([[r, b]]))
    return y_surf


# p = (r, K, c)
# y = (N)
# dN/dt = rN*ln(K/(N+c))
def gompertz(y, t, p):
    #tumor = p[0] * y[0] * math.log(p[1] / (y[0]+p[2]))
    tumor = p[0] * y[0] * math.log(p[1] / y[0])
    return tumor


def test_gompertz(y, t, p):
    cells_ic = y[0]
    r = p[0]
    K = p[1]
    #c = p[2]
    #y_gomp = odeint(gompertz, cells_ic, t, args=tuple([[r, K, c]]))
    y_gomp = odeint(gompertz, cells_ic, t, args=tuple([[r, K]]))
    return y_gomp


# p = (r, b)
# y = (N)
# dN/dt = rN^(2/3) - bN
def bertalanffy(y, t, p):
    tumor = p[0] * y[0] ** (2 / 3) - p[1] * y[0]
    return tumor


def test_bertalanffy(y, t, p):
    cells_ic = y[0]
    r = p[0]
    b = p[1]
    y_bert = odeint(bertalanffy, cells_ic, t, args=tuple([[r, b]]))
    return y_bert



# Time delay on cell death
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

# theta = (p, d, K, np)
# y = (N)
# dN/dt = (p-d)N(1-N/K)
def M3_new(y, t, theta):
    if t < T:
        tumor = (theta[0] - 0) * y[0] * (1 - y[0] / theta[2])
    else:
        tumor = (theta[3] - theta[1]) * y[0] * (1 - y[0] / theta[2])
    return tumor


def test_M3_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    np = theta[3]
    y_M3 = odeint(M3_new, cells_ic, t, args=tuple([[p, d, K, np]]))
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


# theta = (p, d, K, r, np)
# y = (N)
# dN/dt = (p-d*t*e^(1-rt))N(1-N/K)
def M5_new(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1.0 - y[0] / theta[2])
    else:
        tumor = (theta[4] - theta[1] * (t-T) * np.exp(-theta[3] * (t-T))) * y[
            0] * (1 - y[0] / theta[2])
    return tumor


def test_M5_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    np = theta[4]
    y_M5 = odeint(M5_new, cells_ic, t, args=tuple([[p, d, K, r, np]]))
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

# theta = (p, d, K, np)
# y = (N)
# dN/dt = pN(1-N/K)-dN
def M7_new(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * y[0]
    else:
        tumor = theta[3] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0]
    return tumor


def test_M7_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    np = theta[3]
    y_M7 = odeint(M7_new, cells_ic, t, args=tuple([[p, d, K, np]]))
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

# theta = (p, d, K, g, np)
# y = (N)
# dN/dt = pN(1-N/K)-dN/(1+gt)
def M8_new(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                y[0] / (1 + theta[3] * (t-T))
    else:
        tumor = theta[4] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0] / (
                    1 + theta[3] * (t-T))
    return tumor


def test_M8_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    g = theta[3]
    np = theta[4]
    y_M8 = odeint(M8_new, cells_ic, t, args=tuple([[p, d, K, g, np]]))
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

# theta = (p, d, K, r, np)
# y = (N)
# dN/dt = pN(1-N/K)-de^(-rt)N
def M9_new(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                y[0] * np.exp(-theta[3] * (t-T))
    else:
        tumor = theta[4] * y[0] * (1 - y[0] / theta[2]) - theta[1] * y[0] * np.exp(
            -theta[3] * (t-T))
    return tumor


def test_M9_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    np = theta[4]
    y_M9 = odeint(M9_new, cells_ic, t, args=tuple([[p, d, K, r, np]]))
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

# theta = (p, d, K, r)
# y = (N)
# dN/dt = pN(1-N/K)-dte^(-rt)N
def M10_new(y, t, theta):
    if t < T:
        tumor = theta[0] * y[0] * (1 - y[0] / theta[2]) - 0 * \
                theta[3] * y[0] * np.exp(1 - theta[3] * (t-T))
    else:
        tumor = theta[4] * y[0] * (1 - y[0] / theta[2]) - theta[1] *(t-T)* y[
            0] * np.exp(-theta[3] * (t-T))
    return tumor


def test_M10_new(y, t, theta):
    cells_ic = y[0]
    p = theta[0]
    d = theta[1]
    K = theta[2]
    r = theta[3]
    np = theta[4]
    y_M10 = odeint(M10_new, cells_ic, t, args=tuple([[p, d, K, r, np]]))
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


