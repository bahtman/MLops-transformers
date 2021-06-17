import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError('Expected input to a 2D tensor but instead it is: ', x.ndim)
        if x.shape[1] != 768:
            raise ValueError('Expected each sample to have shape [768] but had: ', x.shape )
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = torch.sigmoid(self.fc4(x))

        return x
