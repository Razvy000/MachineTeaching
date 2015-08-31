import os

from matplotlib import pylab
from scipy.stats import entropy
from scipy.stats import norm

import numpy as np


# https://en.wikipedia.org/wiki/Mutual_information
CHART_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")

def mutual_info(x, y, bins=10):
    
    counts_xy, bins_x, bins_y = np.histogram2d(x, y, bins=(bins, bins))
    counts_x, bins = np.histogram(x, bins=bins)
    counts_y, bins = np.histogram(y, bins=bins)
    
    # print x
    # print y
    # print counts_xy, bins_x, bins_y 
    # print counts_x
    # print counts_y

    counts_xy += 1
    counts_x += 1
    counts_y += 1
    P_xy = counts_xy / np.sum(counts_xy, dtype=float)
    P_x = counts_x / np.sum(counts_x, dtype=float)
    P_y = counts_y / np.sum(counts_y, dtype=float)

    I_xy = np.sum(P_xy * np.log2(P_xy / (P_x.reshape(-1, 1) * P_y)))

    return I_xy / (entropy(counts_x) + entropy(counts_y))


def plot_mi_func(x, y):

    mi = mutual_info(x, y)
    title = "NI($X_1$, $X_2$) = %.3f" % mi
    pylab.scatter(x, y)
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")
    # pylab.show()


def plot_mi_demo():

    # determinism!
    np.random.seed(0)
    
    # plot1
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(0, 10, 0.2)

    pylab.subplot(221)
    y = 0.5 * x + norm.rvs(1, scale=.01, size=len(x))
    plot_mi_func(x, y)
    
    pylab.subplot(222)
    y = 0.5 * x + norm.rvs(1, scale=.1, size=len(x))
    plot_mi_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x + norm.rvs(1, scale=1, size=len(x))
    plot_mi_func(x, y)

    pylab.subplot(224)
    y = norm.rvs(1, scale=10, size=len(x))
    plot_mi_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "c_mutual_information_1_ok.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

    ##############
    # plot 2
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(-5, 5, 0.2)

    pylab.subplot(221)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.01, size=len(x))
    plot_mi_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.1, size=len(x))
    plot_mi_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=1, size=len(x))
    plot_mi_func(x, y)

    pylab.subplot(224)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=10, size=len(x))
    plot_mi_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "c_mutual_information_2_ok.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
 
    
if __name__ == '__main__':
    plot_mi_demo()
