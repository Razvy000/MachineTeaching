import numpy as np

def load_dataset(name):
    data = []
    labels = []
    with open('./data/' + name + '.tsv') as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            # label is last
            labels.append(tokens[-1])
            # features
            features = [float(tk) for tk in tokens[:-1]]
            data.append(features)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
