
### Latent Dirichlet allocation (LDA)

# not confuse with Linear Discriminant Analysis

# install pygame, simplejson and pytagcloud
#install gensym for the 
from __future__ import print_function
from wordcloud import create_cloud
from gensim import corpora, models, matutils

import matplotlib.pyplot as plt
import numpy as np
from os import path


NUM_TOPICS = 100

# Download data/ap

# Check
if not path.exists('./data/ap/ap.dat'):
    print("Please donwload the data")

# Load the data
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# Build the topic model
model = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=None)

# Iterate over all the topics in the model
for topic_id in range(model.num_topics):
    how_many_to_show = 64
    words = model.show_topic(topic_id, how_many_to_show)
    tf = sum(f for f,w in words)
    with open('topics.txt', 'w') as output:
        output.write('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for f,w in words))
        output.write("\n\n\n")

# Most discussed topic = highest total weight
topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
topics.shape
# 100, 2246

weight  = topics.sum(1)
weight.shape
# 100,

max_topic = weight.argmax()
# 30
np.sum(topics[30] > 0.1)
# 960 from 2246 topics have 0.1 frecvency or higher for topic 30


# get the top 64 words for this topic
words = model.show_topic(max_topic, 64)

### Create a word cloud using pytagcloud
create_cloud('lda_gensim_tagcloud.png', words)

### Plot number of topics, number of posts
num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('topics_vs_docs1.png')

# change alpha and plot again
# bigger alpha => more topics per document
# DEFAULT gensim ALPHA = 1 / len(corpus)
ALPHA = 1.0

model1 = models.ldamodel.LdaModel( corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=ALPHA)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')

# carefully place description
ax.text(9, 223, r'default alpha=1/len(corpus)')
ax.text(26, 156, 'alpha = 1.0')
fig.tight_layout()
fig.savefig('topics_vs_docs2.png')