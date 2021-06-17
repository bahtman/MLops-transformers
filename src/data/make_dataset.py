# -*- coding: utf-8 -*-
import pickle
import string
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/spam.csv", encoding='latin-1')
df["v1"].replace({"ham": 0, "spam": 1}, inplace=True)

df.rename({"v1": "is_spam", "v2": "message"}, axis=1, inplace=True)


def clean_sentence(s):
    """Given a sentence remove its punctuation and stop words"""
    stop_words = set(stopwords.words('english'))
    # remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(s)
    # removing stop-words
    cleaned_s = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned_s[:10])  # using the first 10 tokens only


df["message"] = df["message"].apply(clean_sentence)

tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

tokenized = df["message"].apply(lambda x:
                                tokenizer.encode(x, add_special_tokens=True))
# get the length of the longest tokenized sentence
max_len = tokenized.apply(len).max()

# padd the rest of the sentence with zeros if
# the sentence is smaller than the longest sentence
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
# create a torch tensor for the padded sentences
input_ids = torch.tensor(padded)
# create a torch tensor for the attention matrix
attention_mask = torch.tensor(attention_mask)


with torch.no_grad():
    encoder_hidden_state = model(input_ids, attention_mask=attention_mask)

X = encoder_hidden_state[0][:, 0, :].numpy()
y = df["is_spam"]

X_train, X_val_test, y_train, y_val_test = train_test_split(X,
                                                            y, test_size = 0.3, stratify=y,
                                                            random_state=17)
X_val, X_test, y_val, y_test = train_test_split(X,
                                                y, test_size = 0.2, stratify=y,
                                                random_state=17)

pickle.dump(X_train, open("data/processed/X_train.pkl", 'wb'))
pickle.dump(X_test, open("data/processed/X_test.pkl", 'wb'))
pickle.dump(y_train, open("data/processed/y_train.pkl", 'wb'))
pickle.dump(y_test, open("data/processed/y_test.pkl", 'wb'))
pickle.dump(X_val, open("data/processed/X_val.pkl", 'wb'))
pickle.dump(y_val, open("data/processed/y_val.pkl", 'wb'))