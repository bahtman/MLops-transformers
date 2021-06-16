import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from model import MyAwesomeModel
from torch import optim

#wandb.init()
batch_size = 128
X_train = pd.read_pickle('../../data/processed/X_train.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')
X_train = torch.tensor(X_train)
y_train = y_train.to_numpy()
y_train = torch.tensor(y_train)

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train, y_train)),
            batch_size=batch_size, shuffle=True)
valloader = 
loss_list = []
val_loss_list = []
print("Training day and night")
model = MyAwesomeModel()
#wandb.watch(model, log_freq=500)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 1000
steps = 0
model.train()
lowest_val_loss = np.inf
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
        #wandb.log({"train_loss": loss})
    else:
        loss_list.append(running_loss/len(trainloader))
        if e % 10 == 0:
            print("at epoch: ",e,f"the Training loss is : {running_loss/len(trainloader)}")
    with torch.no_grad():
        running_loss_val = 0
        model.eval()
        for texts, labels in valloader:
            output = model(texts)
            loss_val = criterion(output.float(), labels.float())
            running_loss_val += loss_val.item()
            #wandb.log({"val_loss": loss_val})
        else:
            val_loss_list.append(running_loss_val/len(valloader))
    if running_loss_val / len(valloader) < lowest_val_loss:
        torch.save(model, '../../models/model.pth')
        lowest_val_loss = running_loss_val/len(valloader)
            
    #wandb.log({"texts": [wandb.Image(i) for i in texts]})
plt.figure()
epoch = np.arange(len(loss_list))
plt.plot(epoch, loss_list)
plt.plot(epoch, )
plt.legend(['Training loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()

