import pandas as pd
import os
import sys
from gensim.models import Doc2Vec, Phrases
from gensim.models.doc2vec import LabeledSentence

pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_train.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_test.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'stars'
X = 'clean_text'
label_id = 'business_id'


df_train[X] = df_train[raw_X].apply(clean_text)


LabeledSentence("hello world".split(), ['123'])




labeled_docs = []
for index, row in df_train.iterrows():
        labeled_docs.append(LabeledSentence(row[X].split(), [row[label_id]]))


# from random import shuffle
# shuffle(labeled_docs)

# ----------------------------------
# Train the model
# ----------------------------------
model = Doc2Vec(dm=1, dbow_words=1, min_count=4, negative=5,
                hs=0, sample=1e-4, window=10, size=500, workers=15)

model.build_vocab(labeled_docs)

# from gensim.models.word2vec import Word2Vec
# model.load_word2vec_format('/home/edward/work/projects/finance/data/GoogleNews-vectors-negative300.bin', binary=True)
model.train(labeled_docs)


# model.save('model.doc2vec')
# model = Doc2Vec.load('model.doc2vec')

# Find words similar to query word
print(model.docvecs.most_similar(positive= ['vx-31tJE_mhf9I0w1E2zaA']))


print(model.most_similar(positive=['taco']))
