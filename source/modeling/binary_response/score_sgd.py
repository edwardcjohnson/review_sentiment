import pandas as pd
import os
import sys
import re
import numpy as np
import xgboost as xgb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import scipy.stats as st
from time import time



#sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) 
pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_test_binary_response.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)

y = 'stars'
y_binary = 'y_binary'
X_raw = 'text'
X = 'clean_text'

pipeline = Pipeline([
    ('vect', CountVectorizer() ),
    ('tfidf', TfidfTransformer() ),
    ('clf', SGDClassifier() )
    ])

parameters = {
    'vect__max_df': 1.0,
    'vect__max_features': None,
    'vect__ngram_range': (1, 2),
    'tfidf__use_idf': True,
    'tfidf__norm': 'l2',
    'clf__alpha':  0.000001,
    'clf__penalty': 'l2',
    'clf__n_iter': 10,
    'clf__loss': 'hinge' # 'log' for logistic, 'hinge' for SVM
}

# Train final model
t0 = time()
pipeline.set_params(**parameters).fit(df_train[X], df_train[y_binary])
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
