
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# get intuition for iris dataset
# load iris dataset
data = load_iris()
# features 
features = data.data
feature_names = data['feature_names']
# classes
target = data['target']
target_names = data.target_names

# plot all possible feature combinations
fig, axes = plt.subplots(2, 3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

for i, (p0, p1) in enumerate(pairs):
    
    ax = axes.flat[i]
    
    for t, marker, c in zip(xrange(3), ">ox", "rgb"):
        # plot 
        ax.scatter(features[target == t, 0], features[target == t, 1], marker=marker, c=c)
    
    ax.set_xlabel(feature_names[p0])
    ax.set_ylabel(feature_names[p1])
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

# save to file
fig.savefig('iris.png')
