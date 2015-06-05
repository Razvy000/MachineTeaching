
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")    
TOY_DIR = os.path.join(DATA_DIR, "toy")

posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

# all data upfront
X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape

print("#samples: %d" % num_samples)
print("#features: %d" % num_features)

print(vectorizer.get_feature_names())
'''
[u'about', u'actually', u'capabilities', u'contains', u'data',
u'databases', u'images', u'imaging', u'interesting', u'is', u'it',
u'learning', u'machine', u'most', u'much', u'not', u'permanently',
u'post', u'provide', u'safe', u'storage', u'store', u'stuff',
u'this', u'toy']
'''

# a new post comes
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

# the vectors from vectorizer are sparse

print(new_post_vec)

# full ndarray
print(new_post_vec.toarray())


### naive similarity measurement
import scipy as sp

def dist_raw(v1,v2):
    delta = v1-v2
    # norm() function calculates the Euclidean norm(shortest distance)
    return sp.linalg.norm(delta.toarray())
    
    
import sys

def best_post(dist_f):
    best_doc = None
    best_dist = sys.maxint
    best_i = None
    for i in range(0, num_samples):
        post = posts[i]    
        if post == new_post:
            continue    
        post_vec = X_train.getrow(i)
        d = dist_f(post_vec, new_post_vec)
        print "=== Post %i with dist=%.2f: %s"%(i, d, post)
        if d < best_dist:
            best_dist = d
            best_i = i
            
    print("Best post is %i with dist=%.2f"%(best_i, best_dist))
    
# get best post using euclidean dist
best_post(dist_raw)
'''
=== Post 0 with dist=4.00: This is a toy post about machine learning.
Actually, it contains not much interesting stuff.
=== Post 1 with dist=1.73: Imaging databases provide storage
capabilities.
=== Post 2 with dist=2.00: Most imaging databases safe images
permanently.
=== Post 3 with dist=1.41: Imaging databases store data.
=== Post 4 with dist=5.10: Imaging databases store data. Imaging
databases store data. Imaging databases store data.
Best post is 3 with dist=1.41
'''


# post 3 and post 4 should be somewhat similar to our new post, post 4 is duplicated 3 times
print(X_train.getrow(3).toarray())
# [[0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]]
print(X_train.getrow(4).toarray())
# [[0 0 0 0 3 3 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0]]

### need to normalize the word count vectors

def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

# get beest post using normalized euclidean dist
best_post(dist_norm)
'''
=== Post 0 with dist=1.41: This is a toy post about machine learning.
Actually, it contains not much interesting stuff.
=== Post 1 with dist=0.86: Imaging databases provide storage
capabilities.
=== Post 2 with dist=0.92: Most imaging databases safe images
permanently.
=== Post 3 with dist=0.77: Imaging databases store data.
=== Post 4 with dist=0.77: Imaging databases store data. Imaging
databases store data. Imaging databases store data.
Best post is 3 with dist=0.77
'''


### Removing less important words
# called stop words, appear everywhere, carry little information
vectorizer = CountVectorizer(min_df=1, stop_words='english')

# usual stop words in english
sorted(vectorizer.get_stop_words())[0:20]
'''
['a', 'about', 'above', 'across', 'after', 'afterwards', 'again',
'against', 'all', 'almost', 'alone', 'along', 'already', 'also',
'''

# 18 words now
len(vectorizer.get_feature_names())

best_post(dist_norm)
'''
=== Post 0 with dist=1.41: This is a toy post about machine learning.
Actually, it contains not much interesting stuff.
=== Post 1 with dist=0.86: Imaging databases provide storage
capabilities.
=== Post 2 with dist=0.86: Most imaging databases safe images
permanently.
=== Post 3 with dist=0.77: Imaging databases store data.
=== Post 4 with dist=0.77: Imaging databases store data. Imaging
databases store data. Imaging databases store data.
Best post is 3 with dist=0.77
'''


### Stemming
# similar words in different variants
# ex: imaging, images
# use Natural Language Toolkit (NLTK)
# link with vectorizer



    