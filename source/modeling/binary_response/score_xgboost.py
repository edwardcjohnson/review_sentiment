import pandas as pd
import os
import sys
import re
import numpy as np
import xgboost as xgb
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBClassifier
import scipy.stats as st
from time import time



#sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) 

pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_train.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_test.pickle'

df_train = pd.read_pickle(pickle_file_train)
df_test = pd.read_pickle(pickle_file_test)


y = 'stars'
y_binary = 'y_binary'
X_raw = 'text'
X = 'clean_text'



pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range = (1, 2), max_features = None) ), # max_features = 20000
    ('tfidf', TfidfTransformer() ),
    ('svd', TruncatedSVD(n_components=300) ),
    ('clf', XGBClassifier(nthread = 5) )
])


parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__n_estimators': 980,
    'clf__max_depth': 10,
    'clf__learning_rate': 0.08,
    'clf__colsample_bytree': 0.94,
    'clf__subsample': 0.9,
    'clf__gamma': 5.2,
    'clf__reg_alpha': 6.2,
    'clf__min_child_weight': 15.6
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
