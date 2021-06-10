import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np


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
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 5
loss_list = []
for e in range(epochs):
    running_loss = 0
    for i in range(len(train_data)):
        optimizer.zero_grad()
        text = train_data.iloc[i]
        label = train_label.iloc[1]
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        labels = tokenizer(label, return_tensors="pt")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        loss_list.append(running_loss/len(train_data))
        print(f"Training loss: {running_loss/len(train_data)}")
plt.figure()
epoch = np.arange(len(loss_list))
print(len(loss_list))
print(epoch)
plt.plot(epoch, loss_list)
plt.legend(['Training loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()
plt.savefig('../../reports/figures/loss_curve')
torch.save(model, '../../models/model.pth')

