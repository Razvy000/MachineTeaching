import os
import sklearn.datasets
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp

### READING POSTS

MLCOMP_DIR = os.path.join(os.path.dirname(__file__), 'data')

print(MLCOMP_DIR)

#data = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root = MLCOMP_DIR)
#data = sklearn.datasets.fetch_20newsgroups(subset="all")
#print("Number of total posts: %i" % len(data.filenames))
#print(data.filenames)
#print(len(data.filenames))
#data.target_names

# choose among training and test sets 
#train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root = MLCOMP_DIR)
#print(len(train_data.filenames))

test_data = sklearn.datasets.load_mlcomp('20news-18828', 'test', mlcomp_root = MLCOMP_DIR)
print(len(test_data.filenames))

# restrict to some newsgroups for simplicity, use categories
groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.load_mlcomp("20news-18828", 'train', mlcomp_root = MLCOMP_DIR, categories=groups)

print(len(train_data.filenames))


### Clustering POSTS

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        
#ignore invalid characters ( UnicodeDecodeError
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, 
                stop_words = 'english', charset_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)

num_samples, num_features = vectorized.shape

print("#samples: %d, #features: %d" % (num_samples, num_features))

# experiment with different sizes
num_clusters = 50

from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1)
clustered = km.fit(vectorized)

# clustering information after fitting
print("km.labels_=%s" % km.labels_)
# km.labels_=[ 6 34 22 ...,  2 21 26]

print("km.labels_.shape=%s" % km.labels_.shape)
# km.labels_.shape=3529

### Metrics
from sklearn import metrics
labels = train_data.target
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# Homogeneity: 0.400
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# Completeness: 0.206
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# V-measure: 0.272
print("Adjusted Rand Index: %0.3f" %  metrics.adjusted_rand_score(labels, km.labels_))
# Adjusted Rand Index: 0.064
print("Adjusted Mutual Information: %0.3f" %  metrics.adjusted_mutual_info_score(labels, km.labels_))
# Adjusted Mutual Information: 0.197
print(("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(vectorized, labels, sample_size=1000)))
# Silhouette Coefficient: 0.006

### Put everything together

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
new_post_vec = vectorizer.transform([new_post])

# the cluster for this new post
new_post_label = km.predict(new_post_vec)[0]

# posts from the same cluster
# (km.labels_ == new_post_label).nonzero() is a tuple, first elem is the list
similar_indices = (km.labels_ == new_post_label).nonzero()[0]  
len(similar_indices)
#47


# get similar
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar)

print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)

