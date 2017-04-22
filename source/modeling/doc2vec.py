import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from collections import Counter, defaultdict
from gensim.models import Doc2Vec, Phrases
from gensim.models.doc2vec import LabeledSentence
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

from time import time


pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_test_binary_response.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'stars'
y_binary = 'y_binary'
X_raw = 'text'
X = 'clean_text'
label_id = 'review_id'


def label_docs(df, text_col, label_col):
    labeled_docs = []
    for index, row in df.iterrows():
        labeled_docs.append(LabeledSentence(row[text_col].split(), [row[label_col]]))
    return labeled_docs


# df_train = df_train[0:10000] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# df_test = df_test[0:10000] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

labeled_docs_train = []
labeled_docs_train = label_docs(df_train, X, label_id)
labeled_docs_test = []
labeled_docs_test = label_docs(df_test, X, label_id)

# from random import shuffle
# shuffle(labeled_docs)

# ----------------------------------
# Train the model
# ----------------------------------

size = 500
model = Doc2Vec(dm=1, dbow_words=1, min_count=4, negative=5,
                hs=0, sample=1e-4, window=10, size=size, workers=15)

model.build_vocab(labeled_docs_train)

# from gensim.models.word2vec import Word2Vec
# model.load_word2vec_format('/home/edward/work/projects/finance/data/GoogleNews-vectors-negative300.bin', binary=True)
model.train(labeled_docs_train)




# model.save('model.doc2vec')
# model = Doc2Vec.load('model.doc2vec')

# Find words similar to query word
print(model.docvecs.most_similar(positive= ['vx-31tJE_mhf9I0w1E2zaA']))


print(model.most_similar(positive=['taco']))


# --------------------------
# Binary Classification
#
# --------------------------

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs = getVecs(model, labeled_docs_train, size)

from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l2')
lr.fit(train_vecs, df_train[y_binary])

prediction = lr.predict(train_vecs)
roc_auc_score(df_train[y_binary], prediction)


# Train on test set
model = Doc2Vec(dm=1, dbow_words=1, min_count=4, negative=5,
                hs=0, sample=1e-4, window=8, size=500, workers=15)
model.build_vocab(labeled_docs_test)
model.train(labeled_docs_test)

test_vecs = getVecs(model, labeled_docs_test, size)

prediction = lr.predict(test_vecs)
accuracy_score(df_test[y_binary], prediction)
roc_auc_score(df_test[y_binary], prediction)


#######################################
w2v = dict(zip(model.index2word, model.syn0))

df_w2v = pd.DataFrame(w2v)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
            ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(iter(word2vec.items()).__next__())
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
                ])


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


pipeline = Pipeline([
    ('word2vec_vectorizer', TfidfEmbeddingVectorizer(w2v)),
    ('clf', SGDClassifier())])

pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_test_binary_response.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'stars'
y_binary = 'y_binary'
X_raw = 'text'
X = 'clean_text'



scores = sorted([(name, cross_val_score(model, X, y, cv=5).mean()) 
                 for name, model in all_models], 
                key=lambda (_, x): -x)




# Train final model
t0 = time()
pipeline.fit(train_vecs, df_train[y_binary])
print("Done in %0.3fs \n" % (time() - t0))

# Return predictions from final model
prediction = pipeline.predict(df_test[X])

# Return metric from final model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
#pipeline.score(df_test[X], df_test[y_binary])

roc_auc_score(df_test[y_binary], prediction)
accuracy_score(df_test[y_binary], prediction)
confusion_matrix(df_test[y_binary], prediction)
print(classification_report(df_test[y_binary], prediction))
