import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Instantiate CSV
df = pd.read_csv('./csv/cannabis.csv')

# Deal with null values (missing descriptions)
# print(df['description'].isnull().sum())
df = df.dropna()

# Turn pandas column into list
text_list = df['description'].tolist()

# Define tokenization function
def tokenize(doc):
    tokens = re.sub('[^a-zA-Z 0-9]', '', doc)
    tokens = tokens.lower().split()
    return tokens

'''
Create document-term matrix
'''
# Instantiate vectorizer object
tfidf = TfidfVectorizer(stop_words = 'english',
                        tokenizer = tokenize,
                        ngram_range = (1,2),
                        min_df = 7,
                        max_df = 0.6)

# Create vocab and tf-idf score
dtm = tfidf.fit_transform(text_list)

# Get feature names to use as df column headers
dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

# Pickle dtm for use in strain_routes.py
with open('./pickle_models/nlp_dtm.pkl', 'wb') as nlp_pkl_file:
    pickle.dump(dtm, nlp_pkl_file)
