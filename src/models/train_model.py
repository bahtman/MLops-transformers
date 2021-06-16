import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from model import MyAwesomeModel
from torch import optim

wandb.init()

X_train = pd.read_pickle('../../data/processed/X_train.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train, y_train)),
            batch_size=64, shuffle=True)

loss_list = []
print("Training day and night")
model = MyAwesomeModel()
wandb.watch(model, log_freq=100)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 20
steps = 0
model.train()
for e in range(epochs):
    running_loss = 0
    for texts, labels in trainloader:
        optimizer.zero_grad()
        output = model(texts)
        output = torch.squeeze(output, 1)
        loss = criterion(output.float(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"loss": loss})
    else:
        loss_list.append(running_loss/len(trainloader))
        print(f"Training loss: {running_loss/len(trainloader)}")
    wandb.log({"texts": [wandb.Image(i) for i in texts]})
plt.figure()
epoch = np.arange(len(loss_list))
plt.plot(epoch, loss_list)
plt.legend(['Training loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()
torch.save(model, '../../models/model.pth')
