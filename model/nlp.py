import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Instantiate CSV
df = pd.read_csv('../data/csv/cannabis.csv')
# df.head()
# print(df.shape)

# Deal with null values (missing descriptions)
# df['description'].isnull().sum()
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

# Show feature matrix as dataframe
# print(dtm.shape)
# dtm.head()

# Create fake description
test_description = ["My name is Bobby. The only kind of cannabis I've used was a light green and I think it came from Northeast Asia. I don't remember what it was called."]

# Fit on DTM with 5 nn
nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
nn.fit(dtm)

# Query data for similar descriptions
new = tfidf.transform(test_description)

# 5 most similar strain descriptions
nn.kneighbors(new.todense())

# Look at one of the matches -- Observation: Also a "light green" color
text_list[5]
