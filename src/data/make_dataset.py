# -*- coding: utf-8 -*-
import re
import string
import torch
import transformers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/raw/spam.csv", encoding='latin-1')
df["v1"].replace({"ham": 0, "spam":1}, inplace=True)

df.rename({"v1": "is_spam", "v2": "message"},axis=1, inplace=True)


def clean_sentence(s):
    """Given a sentence remove its punctuation and stop words"""
    stop_words = set(stopwords.words('english'))
    s = s.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    tokens = word_tokenize(s)
    cleaned_s = [w for w in tokens if w not in stop_words] # removing stop-words
    return " ".join(cleaned_s[:10]) # using the first 10 tokens only

df["message"] = df["message"].apply(clean_sentence)

tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

tokenized = df["message"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
max_len = tokenized.apply(len).max() # get the length of the longest tokenized sentence

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]) # padd the rest of the sentence with zeros if the sentence is smaller than the longest sentence

attention_mask = np.where(padded != 0, 1, 0)
input_ids = torch.tensor(padded)  # create a torch tensor for the padded sentences
attention_mask = torch.tensor(attention_mask) # create a torch tensor for the attention matrix

with torch.no_grad():
    encoder_hidden_state = model(input_ids, attention_mask=attention_mask)

X = encoder_hidden_state[0][:,0,:].numpy()
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)

X_train.to_pickle('../../data/processed/X_train.pkl')
X_test.to_pickle('../../data/processed/X_test.pkl')
y_train.to_pickle('../../data/processed/y_train.pkl')
y_test.to_pickle('../../data/processed/y_test.pkl')