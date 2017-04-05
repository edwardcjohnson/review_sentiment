# review_sentiment
An application of machine learning methods to predict the star rating corresponding to Yelp reviews.


#### Results
| Truncated SVD?  | SGD           | XGBoost |
| :---------------|:--------------|:---------|
| yes             | TBD           | TBD     |
| no              | 0.609         | TBD     |


* Fitting 3 folds for each of 96 candidates, totalling 288 fits
  * [Parallel(n_jobs=20)]: Done 288 out of 288 | elapsed: 433.7min finished
    * done in 27297.536s 

* Best score: 0.609
* Best parameters set:
	* tfidf__norm: 'l2'
	* tfidf__use_idf: True
	* vect__max_df: 1.0
	* vect__max_features: 50000
	* vect__ngram_range: (1, 2)
