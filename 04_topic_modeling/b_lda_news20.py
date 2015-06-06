from __future__ import print_function

import numpy as np
import nltk.corpus
import nltk.stem
from gensim import corpora, models
import sklearn.datasets
from collections import defaultdict
from scipy.spatial import distance



english_stemmer = nltk.stem.SnowballStemmer('english')

stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['from:', 'subject:', 'writes:', 'writes'])

class DirectText(corpora.textcorpus.TextCorpus):
    
    def get_texts(self):
        return self.input
        
    def __len__(self):
        return len(self.input)
    
dataset = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root='./data')
    
texts = dataset.data


texts = [t.decode('utf-8', 'ignore') for t in texts]

# from "ana are mere" => ['ana', 'are', 'mere']
texts = [t.split() for t in texts]

texts = [map(lambda w: w.lower(),t) for t in texts]

# remove words that contain strange characters
texts = [filter(lambda s: not len(set("+-=_.?!()[]>@0123456789") & set(s)), t)   for t in texts]

# remove short words and stop words
texts = [filter(lambda s: (len(s) > 3) and (s not in stopwords), t) for t in texts]

# run words through stemmer
texts = [map(english_stemmer.stem, t) for t in texts]

# make usage dictionary
usage = defaultdict(int)

for t in texts:
    for w in set(t):
        usage[w] += 1

# remove those that appear more than 10%
limit = len(texts) / 10
too_common = [w for w in usage if usage[w] > limit]
too_coomon = set(too_common)
texts = [filter(lambda s: s not in too_common, t) for t in texts]

# create a direct text inherited from text corpus
corpus = DirectText(texts)
dictionary = corpus.dictionary

try:
    dictionary[3]
except:
    pass

# create the lda model
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary.id2token)

# compute the topics
topics = [model[c] for c in corpus]
print(topics[0])

# store all the topics in numpy arrays
# and compute all pairwise distances
dense = np.zeros( (len(topics), 100), float)
for ti,t in enumerate(topics):
    for tj,v in t:
        dense[ti,tj] = v
'''
# compute all pairwise distances 
pairwise = distance.squareform(distance.pdist(dense))

# MemoryError
#pairwise = np.zeros((len(topics), len(topics)), float)

# %xdel testthingy
# %reset out
# %reset array

largest = pairwise.max()
for ti in range(len(topics)):
    pairwise[ti,ti] = largest+1
    
def closest_to(doc_id):
    return pairwise[doc_id].argmin()
'''