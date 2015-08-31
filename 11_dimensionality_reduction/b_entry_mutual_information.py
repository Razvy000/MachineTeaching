import os

from matplotlib import pylab
from scipy.stats import entropy
from scipy.stats import norm

import numpy as np


# https://en.wikipedia.org/wiki/Mutual_information
CHART_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")

def plot_entropy():
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))

    title = "Entropy $H(X)$"
    pylab.title(title)
    pylab.xlabel("$P(X=$coin will show heads up$)$")
    pylab.ylabel("$H(X)$")

    pylab.xlim(xmin=0, xmax=1.1)
    x = np.arange(0.001, 1, 0.001)
    
    # Claude Shannon's information entropy
    # H(X) = - Sum_i( p(Xi) * log(p(Xi)))
    
    y = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    
    pylab.plot(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "b_entropy_coin.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


if __name__ == '__main__':
    plot_entropy()
