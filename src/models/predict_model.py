import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

accuracy_list = []
model = torch.load('models/model.pth')

X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')
X_test = torch.tensor(X_test)
y_test = y_test.to_numpy()
y_test = torch.tensor(y_test)
testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*(X_test, y_test)),
            batch_size=64, shuffle=True)

model.eval()
with torch.no_grad():
    tp=0; tn = 0; fp = 0; fn = 0
    for images, labels in testloader:
        # images, labels = next(iter(testloader))
        ps = torch.exp(model(images.float()))
        #print("ps = ", ps)
        top_p, top_class = ps.topk(1, dim=1)
        #print("top_p = ", top_p, "and top class = ", top_class)
        equals = top_class == labels.view(*top_class.shape)
        for i in range(len(top_class)):
            if top_class[i] == 1:
                if equals[i] == True:
                    tp += 1
                if equals[i] == False:
                    fp += 1
            if top_class[i] == 0:
                if equals[i] == True:
                    tn += 1
                if equals[i] == False:
                    fn += 1
        #print("equals = ", equals)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        accuracy_list.append(accuracy.item()*100)
batch_number = np.arange(len(accuracy_list))
print("tp = ", tp, "fp = ", fp, "tn = ", tn, "fn = ", fn)
print("mean of accuracy = ", np.mean(accuracy_list))
plt.figure()
plt.plot(batch_number, accuracy_list)
plt.legend(['Test set accuacy'])
plt.xlabel('batch_number'), plt.ylabel('Accuacy')
plt.show()
