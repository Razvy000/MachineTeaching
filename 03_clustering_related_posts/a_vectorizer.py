
from sklearn.feature_extraction.text import CountVectorizer

# vectorizer
vectorizer = CountVectorizer(min_df=1)
# mid_df minimum document frecvency, drop word if less

print(vectorizer)
'''
CountVectorizer(analyzer=u'word', binary=False, charset=None,
        charset_error=None, decode_error=u'strict',
        dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
'''

content = ["How to format my hard disk", 
            " Hard disk format problems "]

X = vectorizer.fit_transform(content)

# seem sorted
#  [u'disk', u'format', u'hard', u'how', u'my', u'problems', u'to']
vectorizer.get_feature_names()

print(X.toarray().transpose())

# [[1 1 1 1 1 0 1]
# [1 1 1 0 0 1 0]]
print(X.toarray())

