import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
accuracy_list = []
model = torch.load('../../models/model.pth')

X_test = pd.read_pickle('../../data/processed/X_test.pkl')
y_test = pd.read_pickle('../../data/processed/y_test.pkl')
X_test = torch.tensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.tensor(y_test)
testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_test,y_test)), batch_size=64, shuffle=True)

model.eval()
with torch.no_grad():
    for images, labels in testloader:
        #images, labels = next(iter(testloader))
        ps = torch.exp(model(images.float()))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        accuracy_list.append(accuracy.item()*100)
epoch = np.arange(len(accuracy_list))
print("mean of accuracy = ", np.mean(accuracy_list))
plt.figure()
plt.plot(epoch, accuracy_list)
plt.legend(['Test set accuacy'])
plt.xlabel('Epochs'), plt.ylabel('Accuacy')
plt.show()
