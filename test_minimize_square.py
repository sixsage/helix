import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

min_r0=1.0
max_r0=10.0
nlayers=10
sigma=0.01

# from: https://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/node3.html
# a track is parameterized by a 5dim helix
# here it is expressed relative to an initial angle, phi0, and calculated and some updated angle phi
# hence it sweeps out the track as you vary phi and keep the other parameters fixed
def track(phi, d0,phi0,pt,dz,tanl):
    alpha = 1/2 # 1/cB
    q=1
    kappa = q/pt
    rho = alpha/kappa
    x = d0*np.cos(phi0) + rho*(np.cos(phi0)-np.cos(phi0+phi))
    y = d0*np.sin(phi0) + rho*(np.sin(phi0)-np.sin(phi0+phi))
    z = dz - rho*tanl*phi
    return x,y,z

# For a given phi and track parameters, calculates the distance from a target r02
# used by find_phi to determine the angle that intersects with a fixed-radius circle detector
def dr(phi, r02,d0,phi0,pt,dz,tanl):

    # get the xyz of the track at this phi
    x,y,z = track(phi, d0,phi0,pt,dz,tanl)
    r2=x*x+y*y

    # get the distance from the target r02
    dr = np.fabs(r2-r02)

    return dr

# Find the phi value where the track intersects a given layer r
def find_phi(r0, d0,phi0,pt,dz,tanl):

    # this is lazy, but rather than inverting the equations we just minimize the distance
    res = scipy.optimize.minimize(dr,0,method='Nelder-Mead',args = (r0, d0,phi0,pt,dz,tanl))#, bounds =(0,1.0))

    return res.x[0]

# calculate the chisq between the track defined by the parameters and the spacepoints given
def chisq(params,x,y,z):
    ihit=0
    c2=0
    # find the hits for the track parameters
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)

        # calculate deviation from observed hit
        c2 = c2 + (x0-x[ihit])**2 + (y0-y[ihit])**2 + (z0-z[ihit])**2   # assume equal uncertainty in x,y,z
        ihit = ihit+1

    return c2

def chebyshev(params,x,y,z):
    ihit=0
    c2=0

    # find the hits for the track parameters
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)

        # calculate deviation from observed hit
        c2 = c2 + max(abs(x0-x[ihit]), abs(y0-y[ihit]), abs(z0-z[ihit]))   # assume equal uncertainty in x,y,z
        ihit = ihit+1

    return c2

# find the intersections with the detector layers for these track parameters, add noise
def make_hits(params):
    xs=[]
    ys=[]
    zs =[]
    
    for r0 in np.linspace(min_r0,max_r0,nlayers):
        phi0 = find_phi(r0*r0,*params)
        x0,y0,z0 = track(phi0,*params)
        xs.append(x0+np.random.normal(scale=sigma))
        ys.append(y0+np.random.normal(scale=sigma))
        zs.append(z0+np.random.normal(scale=sigma))


    return xs,ys,zs

def absolute_difference(prediction, actual):
    return np.abs(actual - prediction)

def absolute_sum(prediction, actual):
    return np.abs(prediction) + np.abs(actual)

def percentage_difference(prediction, actual, epsilon=0.0001):
    return 100 * np.abs(prediction - actual) / (np.abs(actual) + epsilon)

def generate_initial_guesses(min_values, max_values, num_points=4):
    initial_guesses = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        initial_guess = min_values + fraction * (max_values - min_values)
        initial_guesses.append(initial_guess)
    return initial_guesses


def fit_params(x, y, z, parameter_bounds, initial_guesses):
    best_params = None
    best_residual = np.inf

    for initial_guess in initial_guesses:
        res = scipy.optimize.minimize(chisq, initial_guess, args=(x, y, z), method='Nelder-Mead', bounds=parameter_bounds)
        if res.fun < best_residual:
            best_residual = res.fun
            best_params = res.x
    return best_params

PATH = 'tracks_100k_updated_non_gaussian_wider.txt'
FUNCTION_TO_TEST = chisq

NUM_POINTS = 200

with open(PATH, 'r') as file:
    content = file.read()
data_points = content.split('EOT')[:NUM_POINTS]

data_points = [dp.strip() for dp in data_points if dp.strip()]
data_points = [dp.split('\n') for dp in data_points]
data_points = [[[float(cell) for cell in row.split(', ')] for row in dp] for dp in data_points]
targets = [dp[0] for dp in data_points]
np_targets = np.array(targets)
target_average = np.mean(np_targets, axis=0)
min_per_column = np.min(np_targets, axis=0)
max_per_column = np.max(np_targets, axis=0)
parameter_bounds = tuple(zip(min_per_column, max_per_column))
inputs = [dp[1:] for dp in data_points]
xs = [[j[0] for j in i]for i in inputs]
ys = [[j[1] for j in i]for i in inputs]
zs = [[j[2] for j in i]for i in inputs]
outputs = []
outputs_mae = []
outputs_abs_sum = []
i = 0
initial_guesses = generate_initial_guesses(min_per_column, max_per_column)
for x, y, z in zip(xs, ys, zs):
    print(i)
    output = fit_params(x, y, z, parameter_bounds, initial_guesses)
    output = np.array(output)
    outputs.append(percentage_difference(output, np_targets[i]))
    outputs_mae.append(absolute_difference(output, np_targets[i]))
    outputs_abs_sum.append(absolute_sum(output, np_targets[i]))
    i += 1
print(outputs)
# np.savetxt("temp.txt", result_percentage)
result_percentage = np.mean(outputs, axis=0)
print(result_percentage)

print('mae')
result_mae = np.mean(outputs_mae, axis=0)
range_per_column = max_per_column - min_per_column
print("ranges")
print(range_per_column)

percent_error = (result_mae / range_per_column) * 100
print("percent errors")
print('mae / range')
print(percent_error)
np.savetxt("temp.txt", percent_error)
# print('smape')
# mask = np.array(outputs_abs_sum) != 0
# print(np.mean(200 * (np.array(outputs_mae)[mask] / np.array(outputs_abs_sum)[mask]), axis=0))
print(np.min(np_targets, axis=0))
print(np.max(np_targets, axis=0))

# print("d0 out of bound")
# print(len(np_targets[np_targets[:, 0] > 0.02]))
# print(len(np_targets[np_targets[:, 0] < 0]))

# print("phi0 out of bound")
# print(len(np_targets[np_targets[:, 1] > 6.28]))
# print(len(np_targets[np_targets[:, 1] < 0]))

# print("pt out of bound")
# print(len(np_targets[np_targets[:, 2] > 200]))
# print(len(np_targets[np_targets[:, 2] < 25]))

# print("dz out of bound")
# print(len(np_targets[np_targets[:, 3] > 2.5]))
# print(len(np_targets[np_targets[:, 3] < -2.5]))

# print("tanl out of bound")
# print(len(np_targets[np_targets[:, 4] > 1.0]))
# print(len(np_targets[np_targets[:, 4] < -1.0]))