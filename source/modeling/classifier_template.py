import pandas as pd
import os
import sys
import re

pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_train.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_test.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'stars'
X = 'clean_text'

# -----------------------------------------------------
#
# Using a pipeline to train a basic model
# w/o hyper-parameter tuning
#
# -----------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('svd', TruncatedSVD(n_components=50) ),
                     ('clf', RandomForestClassifier(n_estimators = 100)) #('clf', MultinomialNB())
                     ])

clf_fit = pipeline.fit(df_train[X], df_train[y])


# evaluate the model
import numpy as np
predicted = clf_fit.predict(df_test[X])
np.mean(predicted == df_test[y])

from sklearn.metrics import confusion_matrix
confusion_matrix(predicted, df_test[y])




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
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way.
# Note: Parameters of pipelines can be set using '__' separated parameter names:
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=20, verbose=1)

t0 = time()
grid_search.fit(df_train[X], df_train[y])
print("done in %0.3fs \n" % (time() - t0))
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))



# -----------------------------------------------------
#
#  XGBoost Randomized Grid Search
#
# -----------------------------------------------------
import xgboost as xgb
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBClassifier
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=100) ),
    ('clf', XGBClassifier(nthread = 10))
])

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
# Note: Parameters of pipelines can be set using '__' separated parameter names:
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__n_estimators': st.randint(50, 100),
    'clf__max_depth': st.randint(3, 15),
    'clf__learning_rate': st.uniform(0.05, 0.4),
    'clf__colsample_bytree': st.beta(10, 1),
    'clf__subsample': st.beta(10, 1),
    'clf__gamma': st.uniform(0, 10),
    'clf__reg_alpha': st.expon(0, 50),
    'clf__min_child_weight': st.expon(3, 50)
}

random_search = RandomizedSearchCV(pipeline, parameters, n_jobs=1, verbose=1)

t0 = time()
random_search.fit(df_train[X], df_train[y])
print("done in %0.3fs \n" % (time() - t0))
print("Best score: %0.3f" % random_search.best_score_) # Best score: 0.601

print("Best parameters set:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# clf__colsample_bytree: 0.88388520775300616
# clf__gamma: 5.0179132016949684
# clf__learning_rate: 0.26320871235601911
# clf__max_depth: 14
# clf__min_child_weight: 53.659697532572245
# clf__n_estimators: 97
# clf__reg_alpha: 24.508120172772827
# clf__subsample: 0.94894647900913398
