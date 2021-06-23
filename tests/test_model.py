import sys
import pandas as pd
import pytest
import torch
sys.path.insert(1,
                '/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/Project/MLops-transformers/src/models')
from model import MyAwesomeModel


model = MyAwesomeModel()
batch_size = 64

X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train, y_train)),
            batch_size=batch_size, shuffle=True)

X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')
X_test = torch.tensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.tensor(y_test)
testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_test, y_test)),
            batch_size=batch_size, shuffle=True)


def test_model(trainloader=trainloader):
    for texts, labels in trainloader:
        assert texts.shape == torch.Size([batch_size, 768])
        output = model(texts.float())
        assert output.shape == torch.Size([batch_size, 1])
        assert labels.shape == torch.Size([64])
        break


def test_raises_warnings():
    model = MyAwesomeModel()
    batch_size = 64
    with pytest.raises(ValueError):
        test1 = torch.randn(768)
        output = model(test1)
    with pytest.raises(ValueError):
        test2 = torch.randn(batch_size, 64)
        output = model(test2)
