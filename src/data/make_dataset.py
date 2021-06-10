# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def SpamDataSet():
    data = pd.read_csv('../../data/raw/spam.csv')
    datasubset = data[['v1','v2']]
    trainset, testset = train_test_split(datasubset, test_size=0.33, random_state=42)
    trainset.to_pickle('../../data/processed/trainset.pkl')
    testset.to_pickle('../../data/processed/testset.pkl')
    return trainset, testset
SpamDataSet()
