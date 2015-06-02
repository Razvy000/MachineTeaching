import numpy as np


def fit_model(k, features, labels):
    ''' Learn a knn model'''
    # no actual model, just save a copy of inputs
    return k, features.copy(), labels.copy()
    
def plurality(xs):
    ''' Find the most common elements in a collection'''
    from collections import defaultdict
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    maxv = max(counts.values())
    for k,v in counts.items():
        if v == maxv:
            return k

def predict(model, features):
    ''' Apply a knn model'''
    k, train_feats, labels = model
    results = []
    for feature in features:
        labeld_dists = []
        
        # compute all distances between current point and trained points
        for train_feat, label in zip(train_feats, labels):
            # frobenium 2 norm sqrt(sum elems^2)
            dist = np.linalg.norm(feature - train_feat)
            labeld_dists.append( (dist, label) )
        
        # sort by distance
        labeld_dists.sort(key=lambda x: x[0])
        
        # choose the first k points
        labeld_dists = labeld_dists[:k]
        
        # label based on plurality
        predicted_label = plurality([label for dist, label in labeld_dists])
        results.append(predicted_label)
        
    return np.array(results)
    
def accuracy(model, features, labels):
    predicts = predict(model, features)
    return np.mean(predicts == labels)