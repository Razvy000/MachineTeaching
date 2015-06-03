
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
