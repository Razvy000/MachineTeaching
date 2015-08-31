

import os

from matplotlib import pylab
import scipy
from scipy.stats import norm
from scipy.stats import pearsonr

import numpy as np

'''
The Pearson correlation coefficient measures the linear relationship between two datasets. Strictly speaking, Pearson’s correlation requires that each dataset be normally distributed. Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.

The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets. The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
'''
print pearsonr([1, 2, 3], [1, 2, 3])
# (1.0, 0.0)

print pearsonr([1, 2, 3], [-1, -2, -3.1])
# (-0.99962228516121843, 0.017498096813278487)

print pearsonr([1, 2, 3], [1, 2, 3.1])
# (0.99962228516121843, 0.017498096813278487)

print pearsonr([1, 2, 3], [1, 20, 6])
# (0.25383654128340477, 0.83661493668227427)


def plot_correlation_func(x, y):

    r, p = pearsonr(x, y)
    title = "Cor($X_1$, $X_2$) = %.3f" % r
    pylab.scatter(x, y)
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    f1 = scipy.poly1d(scipy.polyfit(x, y, 1))
    pylab.plot(x, f1(x), "r--", linewidth=2)
    # pylab.show()
    
def plot_correlation():
    
    # rerun
    
    # plot 1
    np.random.seed(66)
    
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))
    
    x = np.arange(0, 10, 0.2)
    # print x
    
    pylab.subplot(221)
    y = 0.5 * x + norm.rvs(1, scale=0.01, size=len(x))
    # print y
    plot_correlation_func(x, y)
    
    
    pylab.subplot(222)
    y = 0.5 * x + norm.rvs(1, scale=.1, size=len(x))
    plot_correlation_func(x, y)
    

    pylab.subplot(223)
    y = 0.5 * x + norm.rvs(1, scale=1, size=len(x))
    plot_correlation_func(x, y)

    
    pylab.subplot(224)
    y = norm.rvs(1, scale=10, size=len(x))
    plot_correlation_func(x, y)
    
   
    
    # pylab.autoscale(tight=True)
    pylab.grid(True)
    
    filename = "correlation_pearsonr_1_linear_ok.png"
    CHART_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")
    
    if not os.path.exists(CHART_DIR):
        os.mkdir(CHART_DIR)
        
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    
    #########
    
    # plot 2
    
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))
    
    x = np.arange(-5, 5, 0.2)
    
    pylab.subplot(221)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.01, size=len(x))
    plot_correlation_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.1, size=len(x))
    plot_correlation_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=1, size=len(x))
    plot_correlation_func(x, y)

    pylab.subplot(224)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=10, size=len(x))
    plot_correlation_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)
    
    filename = "correlation_pearsonr_2_quad_bad.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    
if __name__ == '__main__':
    plot_correlation()
    