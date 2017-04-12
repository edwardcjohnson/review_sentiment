import pandas as pd
import os
import sys
import re
import numpy as np

#sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir)) 


# file =  '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_academic_dataset_review.json'
# df = pd.read_json(file, lines=True)

pickle_file = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review.pickle'
# df.to_pickle(pickle_file)
df = pd.read_pickle(pickle_file)


y = 'stars'
y_binary = 'y_binary'
X_raw = 'text'
X = 'clean_text'

num_rows = len(df)

df.dtypes
# proportion of each target category
df.groupby([y]).agg(['count'])/num_rows*100

# 1: 13.0 %
# 2:  8.6 %
# 3: 12.4 %
# 4: 24.9 %
# 5: 41.0 %


def clean_text(text):
    """
    Function to process raw text. Processing involves
    removing punctuation, symbols, and any other 
    characters that are not letters or numbers, 
    and converting all words to lower case.
    Args:
        text: a single string e.g. a review
    Returns:
        A single string that has been processed 
    """
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z0-9]", " ", text)
    # Convert to lower case, and split into individual words
    words = letters_only.lower().split()                             
    # Join the words back into a single string.
    return(" ".join(words))



from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2, random_state=8)

# Apply clean_text() to each row under column X in the data frame
df_train[X] = df_train[X_raw].apply(clean_text)
df_test[X] = df_test[X_raw].apply(clean_text)



pickle_file_train = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_train.pickle'
pickle_file_test = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/raw/yelp_review_test.pickle'

df_train.to_pickle(pickle_file_train)
df_test.to_pickle(pickle_file_test)


# --------------------------------
# Create binary response data
#
#    1 if rating is in {4, 5}
#    0 if rating is in {1, 2}
# --------------------------------

df_binary_response = df[df[y] != 3]
df_binary_response[y_binary] = np.where(df_binary_response[y] < 3, 0, np.where(df_binary_response[y] > 3, 1, 99))
df_binary_response.groupby([y_binary]).agg(['count'])/num_rows*100 
# 0: 21.6%
# 1: 65.9%

df_train, df_test = train_test_split(df_binary_response, test_size = 0.2, random_state=8)

pickle_file_train_binary_response = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_train_binary_response.pickle'
pickle_file_test_binary_response = '/nas/isg_prodops_work/ejon9/repos/plb/review_sentiment/data/processed/yelp_review_test_binary_response.pickle'

df_train.to_pickle(pickle_file_train_binary_response)
df_test.to_pickle(pickle_file_test_binary_response)

# 1      13.0
# 2       8.6
# 3      12.4
# 4      24.9
# 5      41.0
