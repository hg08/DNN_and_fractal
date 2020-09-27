#
# Filename: force_distribution.py
#
# Purpose: 
#     Pan Deng's idea,ie., an infinite series is used as the cost function.  
#     If it is continuous and undifferentiable everywhere, the distribution of the  
#     force of each point in its position may be the reason for Levy's flight. 
#
# ref: [1]https://www.baike.com/wiki/可导误区;处处连续处处不可导
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
    f.savefig("a{:2.1f}_b{:3.1f}_force_{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show() 

def plot_prob(dy):
    f=plt.figure()
    hist, bin_edges = np.histogram(dy, bins=20, range=None, weights=None, density=True)
    _ = plt.hist(hist, bins=20)
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:2.1f}_b{:3.1f}_pdf_force_{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show()

# Main function
if "__name__ == __main__":
    a = 0.95
    z = 3.1 
    b = 2*z +1
    nmax = 40
    n_grid = 1000000
    S= 4000000
    x = np.arange(0,S,1) /n_grid
    y = []
    for iterm in x:
        y.append( calc_series(iterm))
    y= np.array(y)

    # After obtaining y
    dy = np.zeros_like(y)

    # Differentation
    for jj in range(np.size(x)):
        if jj==0:
            dy[jj] = -3*y[jj]+4*y[jj+1]-y[jj+2] # see PPT_y_name: Three-point-derivative
        elif jj==np.size(x)-1:
            dy[jj]=y[jj-2]-4*y[jj-1]+3*y[jj] # threepoint formula; 
        else:
            dy[jj]=-y[jj-1]+y[jj+1]
    
    # Plotting
    plot_y(x[:5000],y[:5000])
    plot_prob(dy)
