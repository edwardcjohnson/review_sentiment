import pandas as pd
import os
import sys
import re


pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'y_binary'
X = 'clean_text'



# -----------------------------------------------------
#
# SGDClassifier
# Grid search through the pipeline to find optimal
# hyper-parameter settings
#
# -----------------------------------------------------

from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('vect', CountVectorizer(max_df = 1.0, ngram_range = (1,2) ) ),
    ('tfidf', TfidfTransformer() ),
    ('clf', SGDClassifier() )
    ])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way.
# Note: Parameters of pipelines can be set using '__' separated parameter names:
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=10, scoring='roc_auc', verbose=1)

t0 = time()
grid_search.fit(df_train[X], df_train[y])
print("done in %0.3fs \n" % (time() - t0))
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

 
