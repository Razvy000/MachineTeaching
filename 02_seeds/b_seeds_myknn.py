import numpy as np
from load import load_dataset
from knn import fit_model, accuracy


features, labels = load_dataset('seeds')

def oneFold():
    # masks for training and testing
    training = np.ones(len(features), bool)
    # sample
    training[1::4] = 0
    testing = ~training

    k = 1
    model = fit_model(k, features[training], labels[training])
    accr = accuracy(model, features[testing], labels[testing])
    print 'Aprox Accuracy was{0:.1%}'.format(accr)


def cross_validate(features, labels):
    
    k = 1
    accr = 0.0
    nFolds = 10
    for fold in range(nFolds):
        training = np.ones(len(features), bool)
        # unsample every nFold
        training[fold::nFolds] = 0
        testing = ~training
        model = fit_model(k, features[training], labels[training])
        accr += accuracy(model, features[testing], labels[testing])
    return accr / nFolds

acc = cross_validate(features, labels)
print('Ten fold cross-validation accuracy was {0:.1%}'.format(acc))

# Z-score (whiten) the features
# 0 axis
features -= features.mean(0)    
features /= features.std(0)

acc = cross_validate(features, labels)
print('Ten fold cross-validation with Z-scoring accuracy was {0:.1%}'.format(acc))