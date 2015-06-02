
from sklearn.datasets import load_iris

# classify by hand


# load iris dataset
data = load_iris()
# features 
features = data.data
feature_names = data.feature_names
# classes
target = data.target
target_names = data.target_names

# make array of string names
labels = target_names[target]

# petal length is feature at pos 2
petal_len = features[:,2]

# check is_setosa
is_setosa = (labels == 'setosa')

# check they are linearly separable
max_setosa = petal_len[is_setosa].max()
min_non_setosa = petal_len[~is_setosa].min()

print('Max of setosa: {0}'.format(max_setosa))
print('Min of others: {0}'.format(min_non_setosa))
# output
# Max of setosa: 1.9
# Min of others: 3.0

# clearly linearly separable

def is_setosa_test(examples):
    if examples[2] < 2.5: print 'Iris Setosa'
    else: print 'Iris Virginica or Iris Versicolour'


is_setosa_test([5.1, 3.5, 1.4, 0.2])
is_setosa_test([5, 2, 3.5, 1])

