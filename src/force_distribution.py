#
# Filename: force_distribution.py
#
# Purpose: 
#     Chen Guozhang and Pan Deng's idea,ie., an infinite series is used as the cost function.  
#     If it is continuous and undifferentiable everywhere, the distribution of the  
#     force of each point in its position may be the reason for Levy's flight. 
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
    #plt.xscale('log')
    plt.plot(x,y)
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:d}_force_S{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show() 

def plot_prob(dy):
    f=plt.figure()
    num_bins = 101
    n, bins, patches = plt.hist(dy,bins=num_bins,density=False,align='mid',histtype='stepfilled',facecolor='g',alpha=0.75)
    plt.title(r"nmax={:3d};$\Delta t$={:7.5f};step={:d}".format(nmax,1/n_grid,S))
    f.savefig("a{:3.2f}_b{:d}_pdf_force_S{:d}_grid{:d}_nmax{:d}.png".format(a,b,S,n_grid,nmax))
    plt.show()

# Main function
if "__name__ == __main__":
    a = 0.9
    z = 3 
    b = 2*z +1
    nmax = 3 
    n_grid = 1000
    S= 40000
    x = np.arange(0,S,1) /n_grid
    dy = []
    for iterm in x:
        dy.append( diff_calc_series(iterm))
    dy= np.array(dy)

    # Plotting
    np.save('dy_a{:3.2f}_b{:d}_force_S{:d}_grid{:d}_nmax{:d}.npy'.format(a,b,S,n_grid,nmax),dy)
    plot_y(x[:5000],dy[:5000])
    plot_prob(dy)
