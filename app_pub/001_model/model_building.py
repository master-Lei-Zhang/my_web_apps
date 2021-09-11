import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import nltk, re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# prepare NLTK text processer
stop = stopwords.words('english')
stop += ['review', 'application', 'chapter', 'recent', 'challenge', 'future', 
         'progress', 'perspective', 'study', '19', 'opportunity', 'state', 'art', 'current']

wl = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def process_text(texts):
    """
    This function uses NLTK techniques to clean text words, very good implementation!
    It is suggested to apply to each row of text in a dataframe.
    Example: process_text(df['title'].tolist())
    """
    
    final_text_list = []
    for sent in texts:
        
        if isinstance(sent, str) == False:
            sent = ''
            
        filtered_sentence = []
        
        sent = sent.lower()
        sent = sent.strip()
        sent = re.sub("\s+", " ", sent)
        sent = re.compile("<.*?>").sub("", sent)
        
        for w in word_tokenize(sent):
            if (not w.isnumeric()) and (len(w) > 2) and (w not in stop):
                filtered_sentence.append(w)
        final_string = " ".join(filtered_sentence)
        
        lemmatized_sentence = []
        words = word_tokenize(final_string)
        word_pos_tags = nltk.pos_tag(words)
        for idx, tag in enumerate(word_pos_tags):
            lemmatized_sentence.append(wl.lemmatize(tag[0], get_wordnet_pos(tag[1])))

        lemmatized_text = " ".join(lemmatized_sentence)

        final_text_list.append(lemmatized_text)
        
    return final_text_list


# define topN recommender
def recommend(target_text, N, tfv, word_vector):
    """
    This function take target text data as input.
    Vectorize it using fitted tfv.
    Calculate cosine similarity with database word_vector.
    Recommend top N literature.
    """
    clean_target = process_text([target_text])
    target_vector = tfv.transform(clean_target).toarray()
    similarity = (1 - pairwise_distances(target_vector, word_vector, metric = 'cosine')).reshape(-1)
    topN_idx = similarity.argsort()[-N:][::-1]

    return topN_idx


# load literature database
df = pd.read_csv('../002_flaskapp/elsevier_2019_2021.csv')

# add a new column with processed title
clean_word = process_text(df['title'].tolist())

# word 2 vector
tfv = TfidfVectorizer(max_features=300, stop_words=stop)
tfv.fit(clean_word)
word_vector = tfv.transform(clean_word).toarray()

with open('model.pkl', 'wb') as f:
    pickle.dump(df, f)
    pickle.dump(tfv, f)
    pickle.dump(word_vector, f)