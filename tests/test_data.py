import torch
import numpy as np
import pytest
import pandas as pd
#train_data, train_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/Project/MLops-transformers/data/processed/training.pt')
#trainset = torch.utils.data.TensorDataset(*trainset)
#test_data, test_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/Project/MLops-transformers/data/processed/test.pt')

X_train = pd.read_pickle('../data/processed/X_train.pkl')
y_train = pd.read_pickle('../data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

X_test = pd.read_pickle('../data/processed/X_test.pkl')
y_test = pd.read_pickle('../data/processed/y_test.pkl')
X_test = torch.tensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.tensor(y_test)
train_data = torch.utils.data.TensorDataset(*(X_train,y_train))
test_data = torch.utils.data.TensorDataset(*(X_test,y_test))



#testset = torch.utils.data.TensorDataset(*testset)
@pytest.mark.parametrize("test_input,expected", [("train_data", torch.Size([4179,2,768])), ("test_data", torch.Size([1393,2,768]))])
def test_data_shape(test_input, expected):
    assert eval(test_input)[0].shape == expected


@pytest.mark.parametrize("test_input,expected", [("train_label", np.arange(2)), ("test_label", np.arange(2))])
def test_label_values(test_input, expected):
    assert np.array_equal(np.unique(eval(test_input)), expected)
