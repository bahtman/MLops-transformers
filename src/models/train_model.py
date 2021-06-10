import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

train_set = pd.read_pickle('../../data/processed/trainset.pkl')
test_set = pd.read_pickle('../../data/processed/testset.pkl')

train_data = train_set['v2']
train_label = train_set['v1']

test_data = test_set['v2']
test_label = test_set['v1']

#train_data_tensor = torch.tensor(train_data)

#trainloader = torch.utils.data.DataLoader(
    #torch.utils.data.TensorDataset(*(train_data,train_label)), batch_size=64, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)