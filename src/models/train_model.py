import torch
import pandas as pd
train_set = pd.read_pickle('../../data/processed/trainset.pkl')
test_set = pd.read_pickle('../../data/processed/testset.pkl')

train_data = train_set['v2']
train_label = train_set['v1']

test_data = test_set['v2']
test_label = test_set['v1']