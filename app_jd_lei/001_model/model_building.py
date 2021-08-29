import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('jd_sure_iter-1.csv')
kw = pd.read_csv('keywords.csv')
words = kw.words.tolist()
vocabulary = {}

for i in range(len(words)):
    vocabulary[words[i]] = i

vectorizer = TfidfVectorizer(vocabulary=vocabulary)

X = df["description"]
y = df.label
weight = 'balanced'

model = Pipeline([('cv', vectorizer),
                  ('lr', LogisticRegression(penalty="l1",
                                            C=10,
                                            solver='saga',
                                            #solver = 'liblinear',
                                            # solver='newton-cg',
                                            multi_class='auto',
                                            class_weight=weight))])

model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)