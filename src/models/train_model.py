import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
#import wandb
import time
from model import MyAwesomeModel
from torch import optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#wandb.init()
batch_size = 128
X_train = torch.tensor(pd.read_pickle('data/processed/X_train.pkl'))
y_train = torch.tensor(pd.read_pickle('data/processed/y_train.pkl').to_numpy())
X_val = torch.tensor(pd.read_pickle('data/processed/X_val.pkl'))
y_val = torch.tensor(pd.read_pickle('data/processed/y_val.pkl').to_numpy())

trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_train, y_train)),
            batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_val, y_val)),
            batch_size=batch_size, shuffle=False)
loss_list = []
val_loss_list = []
res = []

print("Training day and night")
model = MyAwesomeModel()
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids = [0])

#wandb.watch(model, log_freq=500)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 100
steps = 0
model.train()
lowest_val_loss = np.inf
for e in range(epochs):
    start = time.time()
    running_loss = 0
    for texts, labels in trainloader:
        texts, labels = texts.to(device), labels.to(device)
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
        if e % 20 == 0:
            print("at epoch: ",e,f"the Training loss is : {running_loss/len(trainloader)}") 
    with torch.no_grad():
        running_loss_val = 0
        model.eval()
        for texts, labels in valloader:
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts)
            output = torch.squeeze(output,1)
            loss_val = criterion(output.float(), labels.float())
            running_loss_val += loss_val.item()
            #wandb.log({"val_loss": loss_val})
        else:
            val_loss_list.append(running_loss_val/len(valloader))
            if e % 20 == 0:
                print("at epoch: ",e,f"the Validation loss is : {running_loss_val/len(valloader)}") 
    if (running_loss_val / len(valloader)) < lowest_val_loss:
        torch.save(model, 'models/model.pth')
        lowest_val_loss = running_loss_val/len(valloader)
    else:
        continue      
    #wandb.log({"texts": [wandb.texts(i) for i in texts]})
    end = time.time()
    res.append(end - start)
res = np.array(res)
print('Timing:', np.mean(res),'+-',np.std(res))
plt.figure()
epoch = np.arange(len(loss_list))
plt.plot(epoch, loss_list)
plt.plot(epoch, val_loss_list)
plt.legend(['Training loss and validation loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()

