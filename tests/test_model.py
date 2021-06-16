import sys

import pandas as pd
import pytest
import torch
from model import MyAwesomeModel

sys.path.insert(1, 
    '/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/Project/MLops-transformers/src/models')
model = MyAwesomeModel()
batch_size = 64

X_train = pd.read_pickle('../../data/processed/X_train.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train,y_train)), batch_size=batch_size, shuffle=True)

X_test = pd.read_pickle('../../data/processed/X_test.pkl')
y_test = pd.read_pickle('../../data/processed/y_test.pkl')
X_test = torch.tensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.tensor(y_test)
testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_test,y_test)), batch_size=batch_size, shuffle=True)

def test_model(trainloader = trainloader):
    for images, labels in trainloader:
        assert images.shape == torch.Size([batch_size,1,28,28])
        output = model(images.float())
        assert output.shape == torch.Size([batch_size, 10])
        break

def test_raises_warnings():
    model = MyAwesomeModel()
    batch_size = 64
    with pytest.raises(ValueError):
        test1 = torch.randn(1,28,28)
        output = model(test1)
    with pytest.raises(ValueError):
        test2 = torch.randn(batch_size,3,28,28)
        output = model(test2)