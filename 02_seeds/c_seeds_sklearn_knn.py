from __future__ import print_function
import numpy as np
from load import load_dataset
# sklearn implementation of knn
from sklearn.neighbors import KNeighborsClassifier

# load data
features, labels = load_dataset('seeds')
    
def leave_one_out():

    # create a sklearn knn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    
    n = len(features)
    correct = 0.0

    # leave-one-out training
    for ignorefeat in range(n):
        training = np.ones(n, bool)    
        # leave out
        training[ignorefeat] = 0
        testing = ~training
        
        # fit
        classifier.fit(features[training], labels[training])
        
        # predict
        prediction = classifier.predict(features[ignorefeat])
        
        # sum correct
        correct += (prediction == labels[ignorefeat])

    correct /= n

    print('Leave-one-out {}'.format(correct))
 
def use_kfold():
    from sklearn.cross_validation import KFold
    
    # create a sklearn knn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    
    # mean for each fold
    means = []
    
    # kf is a generator of pairs (training, testing) so each iteration is a different fold
    kf = KFold(len(features), n_folds = 3, shuffle=True)
    
    for training, testing in kf:
        # learn
        classifier.fit(features[training], labels[training])
        # apply
        prediction = classifier.predict(features[testing])
        
        # fraction correct
        curmean = np.mean(prediction == labels[testing])
        means.append(curmean)
    print('Cross-validation using KFold: {0}= {1}'.format(means, np.mean(means)))

def use_cross_val_score():
    from sklearn.cross_validation import cross_val_score
    
    # create a sklearn knn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    
    crossed = cross_val_score(classifier, features, labels)
    
    print('Cross-validation using cross_val_score: {0}= {1}'.format(crossed, np.mean(crossed)))

def do_prescale():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import cross_val_score
    
    # create a sklearn knn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    
    # create a pipeline with prescaler + classifier
    classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
    
    crossed = cross_val_score(classifier, features, labels)
    
    print('Prescaler: {0}= {1}'.format(crossed, np.mean(crossed)))
    
def do_confusion():
    ''' generate and print a cross-validated confusion matrix'''
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import KFold
    from sklearn.metrics import confusion_matrix
    
    # load data
    features, labels = load_dataset('seeds')
    
    # create a sklearn knn classifier
    classifier = KNeighborsClassifier(n_neighbors = 4)
    
    # create a pipeline with prescaler + classifier
    classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
    
    kf = KFold(len(features), n_folds = 3, shuffle=True)
    
    names = list(set(labels))
    labels = np.array([names.index(ell) for ell in labels])
    preds = labels.copy()
    preds[:] = -1
    for train, test in kf:
        classifier.fit(features[train], labels[train])
        preds[test] = classifier.predict(features[test])
        
    cmat = confusion_matrix(labels, preds)
    print('Confusion matrix [rows represent true outcome, columns = predicted outcome]')
    print(cmat)
    
    # the explicit float() conversion is necessary in Python 2 (otherwise, result is rounded to 0)
    acc = cmat.trace()/float(cmat.sum())
    print('Accuracy: {0:.1%}'.format(acc))

leave_one_out()
use_kfold()
use_cross_val_score()
do_prescale()
do_confusion()