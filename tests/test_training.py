# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import torch
import wandb
import pytest
from model import MyAwesomeModel
from torch import optim

wandb.init()
batch_size = 64
X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train, y_train)),
            batch_size=batch_size, shuffle=True)

loss_list = []
print("Training day and night")
model = MyAwesomeModel()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 2
steps = 0
@pytest.mark.parametrize("test_input,expected",
                       [("texts", torch.Size([768])),
                        ("output", torch.Size([1]))])
def test_training_script(test_input, expected):
    model.train()
    for e in range(epochs):
        running_loss = 0
        for texts, labels in trainloader:
            optimizer.zero_grad()
            output = model(texts)
            assert eval(test_input)[0].shape == expected
            output = torch.squeeze(output, 1)
            loss = criterion(output.float(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            loss_list.append(running_loss/len(trainloader))
            print(f"Training loss: {running_loss/len(trainloader)}")
