#
# Filename: gd.py
#
# Purpose: 
#     Chen Guozhang's idea,ie.,if we model a sgd and then calculate the distribution of gradients, 
#     of the loss function, it will be meanful. 
#     If the loss is continuous and undifferentiable everywhere, the distribution of the  
#     "real biased forces" may be the reason for Levy's flight. 
#
# ref: [1]https://en.wikipedia.org/wiki/Weierstrass_function  处处连续处处不可导
#      [2]https://www.thinbug.com/q/47204122 
#      [3]https://www.mathworks.com/matlabcentral/fileexchange/73273-fractal-landscape-generator
#      Date         Author           Version
#      ========     =========        =======
#      2020-9-27    GangHuang        Original

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Functions
def zero_to_infinity(nmax=20):
    """yield creates a generator."""
    i = 0
    while True:
        yield i
        i += 1
        if i > nmax:
           break

def calc_series(x):
    res, temp = 0, 0
    for i in zero_to_infinity(nmax):
        res +=  (a**i)*np.cos((b**i) * np.pi * x)
        if res == temp:
            break
        temp = res
    return res

def diff_calc_series(x):
    res, temp = 0, 0
    for i in zero_to_infinity(nmax):
        res +=  -(a**i) * (b**i)* (np.pi) * np.sin((b**i) * np.pi * x)
        if res == temp:
            break
        temp = res
    return res

def plot_y(x,y):
    f=plt.figure()
    plt.plot(x,y)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$Loss$')
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:d}_loss_S{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    #plt.show() 

def plot_traj(x,y_traj):
    f=plt.figure()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$Loss$')
    plt.plot(x,y_traj)
    plt.title(r"Loss evolution: $\eta$={:4.3f};nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(eta,nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:d}_loss_traj_S{:d}_grid{:d}_nmax{:d}_eta{:4.3f}.png".format(a,b,S,n_grid,nmax,eta))
    #plt.show() 

def plot_prob(dy):
    f=plt.figure()
    num_bins = 101
    plt.xlabel(r'gradients ($\nabla x$)')
    plt.ylabel(r'Probability of $\nabla x$')
    n, bins, patches = plt.hist(dy,bins=num_bins,density=False,align='mid',histtype='stepfilled',facecolor='g',alpha=0.75)
    plt.title(r"$\eta$={:4.3f}; nmax={:3d};$\Delta t={:5.4f}$;step={:d}".format(eta,nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:d}_sgd_pdf_force_S{:d}_grid{:d}_nmax{:d}_eta{:4.3f}.png".format(a,b,S,n_grid,nmax,eta))
    #plt.show()

# Main function
if "__name__ == __main__":
    a = 0.9
    z = 3 
    b = 2*z +1
    nmax = 30 
    n_grid = 100000
    S= 2000000
    x = np.arange(0,S,1) /n_grid
    #load the loss function
    y = np.load('a{:3.2f}_b{:d}_loss_S{:d}_grid{:d}_nmax{:d}.npy'.format(a,b,S,n_grid,nmax) )
   
    #To show the loss 
    plot_y(x[:200000],y[:200000])

    # sgd
    n_iterations = 80000

    for eta in [0.01,0.015,0.02,0.05,0.1]:  # learning rate
        #random initialization
        theta = np.random.uniform(0,S,1) # generate one number in the range [0,S).
        theta = theta/n_grid
    
        dy = []
        x_iteration =np.arange(n_iterations)
        y_traj = [] # to record the trajectory of the loss
        for iteration in range(n_iterations):
            gradients = diff_calc_series(theta) 
            dy.append(gradients)
            theta = theta - eta * gradients # updating the x: delta_x = (-gradients) * eta
            y_traj.append(calc_series(theta))
        dy= np.array(dy)
        # Plotting
        np.save('dy_a{:3.2f}_b{:d}_sgd_pdf_S{:d}_grid{:d}_nmax{:d}_eta{:4.3f}.npy'.format(a,b,S,n_grid,nmax,eta),dy)
        plot_prob(dy)
        plot_traj(x_iteration, y_traj)
