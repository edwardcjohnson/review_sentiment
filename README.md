# review_sentiment
An application of machine learning methods to predict the star rating corresponding to Yelp reviews.


## Results
### Multi-label
| Truncated SVD?  | SGD           | XGBoost |
| :---------------|:--------------|:---------|
| yes             | TBD           | TBD     |
| no              | 0.609         | TBD     |

#### CV
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
	
#### Test set
#### SGD
accuracy = 0.6757

 confusion matrix
 
|          | True 1     | True 2     | True 3 | True 4  | True 5 |
| :--------|:-----------|:-----------|:--------|:-------|:-------|
| Pred 1   | 98898      | 3356       |   2808  | 1147   | 2372   |
| Pred 2   | 30192      | 14268      |18716    | 5920   | 2730   |
| Pred 3   | 7830       | 7128       | 39343   | 39461  | 9492   |
| Pred 4   | 2253       | 2060       | 11179   | 103563 | 87068  |
| Pred 5   | 2253       | 567        | 1572    | 31701  | 305195 |


 classification report
 
|          | precision  | recall   | f1-score| support|
|:---------|:-----------|:---------|:--------|:-------| 
| 1        | 0.70       | 0.91     | 0.79    | 108581 |
| 2        | 0.52       | 0.20     | 0.29    | 71826  |
| 3        | 0.53       | 0.38     | 0.44    | 103254 |
| 4        | 0.57       | 0.50     | 0.53    | 206123 |
| 5        | 0.75       | 0.90     | 0.82    | 340846 |
|avg/total | 0.65       | 0.68     | 0.65    | 830630 |

### Binary label
* SGD:
* Fitting 3 folds for each of 48 candidates, totalling 144 fits
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



* XGBoost:
* Fitting 3 folds for each of 10 candidates, totalling 30 fits
  * [Parallel(n_jobs=5)]: Done  30 out of  30 | elapsed: 727.0min finished
    * done in 60602.505s 

* Best score (AUC): 0.979
* Best parameters set:
	* clf__colsample_bytree: 0.93882416583162387
	* clf__gamma: 5.2248856032270332
	* clf__learning_rate: 0.079260905486493102
	* clf__max_depth: 10
	* clf__min_child_weight: 15.596298451557294
	* clf__n_estimators: 981
	* clf__reg_alpha: 6.1698510973535718
	* clf__subsample: 0.90673496002706067*

#### Test Set
#### SGD
* AUC
  * 0.96
* accuracy
  * 0.9726
* confusion_matrix
* array([[168309,  11536],
       [  8349, 538963]])
* classification_report
             precision    recall  f1-score   support

          0       0.95      0.94      0.94    179845
          1       0.98      0.98      0.98    547312

avg / total       0.97      0.97      0.97    727157

