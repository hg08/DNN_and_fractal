#
# Filename: generate_loss.py
#
# Purpose: 
#     Pan Deng's idea,ie., an infinite series is used as the cost function.  
#     If it is continuous and undifferentiable everywhere, the distribution of the  
#     force of each point in its position may be the reason for Levy's flight. 
# Output:
#     Generate the loss, deonted by y.
# ref:[1]https://www.baike.com/wiki/可导误区;处处连续处处不可导
#     [2]https://www.thinbug.com/q/47204122 
#     Date         Author           Version
#     ========     =========        =======
#     2020-9-27    GangHuang        Original

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

def plot_y(x,y):
    f=plt.figure()
    #plt.xscale('log')
    plt.plot(x,y)
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:3.2f}_force_{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show() 

def plot_prob(dy):
    f=plt.figure()
    hist, bin_edges = np.histogram(dy, bins=200, range=(0,100.0), density=True)
    n, bins, patches = plt.hist(hist, bins=200)
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:3.2f}_pdf_force_{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show()

# Main function
if "__name__ == __main__":
    a = 0.9
    z = 3 
    b = 2*z +1
    nmax = 50
    n_grid = 1000
    S= 4000000
    x = np.arange(0,S,1) /n_grid
    y = []
    for iterm in x:
        y.append( calc_series(iterm))
    y= np.array(y)

    #save the loss as an array.
    np.save('a{:3.2f}_b{:d}_loss_S{:d}_grid{:d}_nmax{:d}.npy'.format(a,b,S,n_grid,nmax),y)
    print('The loss array is written in: a{:3.2f}_b{:d}_loss_S{:d}_grid{:d}_nmax{:d}.npy.'.format(a,b,S,n_grid,nmax))
