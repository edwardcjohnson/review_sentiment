# review_sentiment
An application of machine learning methods to predict the star rating corresponding to Yelp reviews.


## Results
#### Multi-label
| Truncated SVD?  | SGD           | XGBoost |
| :---------------|:--------------|:---------|
| yes             | TBD           | TBD     |
| no              | 0.609         | TBD     |


* Fitting 3 folds for each of 96 candidates, totalling 288 fits
  * [Parallel(n_jobs=20)]: Done 288 out of 288 | elapsed: 433.7min finished
    * done in 27297.536s 

* Best score (Accuracy): 0.609
* Best parameters set:
	* tfidf__norm: 'l2'
	* tfidf__use_idf: True
	* vect__max_df: 1.0
	* vect__max_features: 50000
	* vect__ngram_range: (1, 2)

#### Binary label

*Fitting 3 folds for each of 48 candidates, totalling 144 fits
  * [Parallel(n_jobs=10)]: Done  30 tasks      | elapsed: 55.6min
  * [Parallel(n_jobs=10)]: Done 144 out of 144 | elapsed: 223.8min finished
    * done in 13895.965s

* Best score (AUC): 0.994 
* Best parameters set:
        * clf__alpha: 1e-06
        * clf__n_iter: 10
        * clf__penalty: 'l2'
        * tfidf__norm: 'l2'
        * vect__max_features: None


