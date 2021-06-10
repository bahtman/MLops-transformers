import torch
import pandas as pd
from model import MyAwesomeModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

X_train = pd.read_pickle('../../data/processed/X_train.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')


loss_list = []
print("Training day and night")
model = MyAwesomeModel()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 5
steps = 0
model.train()
for e in range(epochs):
    running_loss = 0
    for texts, labels in X_train, y_train:
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        loss_list.append(running_loss/len(trainloader))
        print(f"Training loss: {running_loss/len(trainloader)}")
plt.figure()
epoch = np.arange(len(loss_list))
print(len(loss_list))
print(epoch)
plt.plot(epoch, loss_list)
plt.legend(['Training loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()
torch.save(model, 'model.pth')
