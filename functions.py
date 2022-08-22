# Python libraries
import numpy as np
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

path = "C:/Users/moumouni/Desktop/OC_P5_ml/Stack_questions_cleaned.csv"
data = pd.read_csv(path, sep=';')
from ast import literal_eval

for col in ['Title', 'Body', 'Tags', 'text_comb']:
    data[col] = data[col].apply(literal_eval)


def inv_transform(label):
   """ return the inverse of MultiLabelBinarizer fitted on Tags data
       of the classes in label"""
   # threshold of
   threshold = 0.1
   label  = (label>threshold)*1
   mlb = MultiLabelBinarizer()
   mlb.fit(data.Tags)
   y_mlb = mlb.transform(data.Tags)
   tags = mlb.inverse_transform((label>0.12)*1)
   return tags
#__________________

def get_wordnet_pos(word):
   """Map POS tag to first character lemmatize() accepts"""
   tag = nltk.pos_tag([word])[0][1][0].upper()
   tag_dict = {"J": wordnet.ADJ,
               "N": wordnet.NOUN,
               "V": wordnet.VERB,
               "R": wordnet.ADV}

   return tag_dict.get(tag, wordnet.NOUN)


def sentence_cleaner(text):
   """Function clean text by removing extra spaces, unicode characters,
       English contractions, links, punctuation and numbers.
   """
   # Make text lowercase
   text = text.lower()
   # Remove English contractions
   text = re.sub("\'\w+", ' ', text)
   #
   text = text.encode("ascii", "ignore").decode()
   # Remove ponctuation (except # and ++ for c# and c++)
   text = re.sub('[^\\w\\s#\\s++]', ' ', text)
   # Remove numbers
   text = re.sub(r'\w*\d+\w*', ' ', text)
   # Remove extra spaces
   text = re.sub('\s+', ' ', text)
   text = BeautifulSoup(text, "lxml").text
   # Tokenization
   text = text.split()

   # List of stop words from NLTK
   stop_words = stopwords.words('english')
   # Remove stop words
   text = [word for word in text if word not in stop_words
           and len(word) > 2]
   # Lemmatizer
   lemm = WordNetLemmatizer()
   text = [lemm.lemmatize(w, get_wordnet_pos(w)) for w in text]

   # Return cleaned text
   return text
