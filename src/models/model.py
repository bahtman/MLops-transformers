import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.fc1 = nn.Linear(768, hparams['hidden_dim_1'])
        self.fc2 = nn.Linear(hparams['hidden_dim_1'], hparams['hidden_dim_2'])
        self.fc3 = nn.Linear(hparams['hidden_dim_2'], hparams['hidden_dim_3'])
        self.fc4 = nn.Linear(hparams['hidden_dim_3'], 1)

        # Dropout module with 0.2 drop probability
        self.dropout_input = nn.Dropout(p=hparams['dropout_input'])
        self.dropout_hidden = nn.Dropout(p=hparams['dropout_hidden'])

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError('Expected input to a 2D tensor but instead it is: ',
                            x.ndim)
        if x.shape[1] != 768:
            raise ValueError('Expected each sample to have shape [768] but had: ',
                            x.shape)
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout_input(F.relu(self.fc1(x)))
        x = self.dropout_hidden(F.relu(self.fc2(x)))
        x = self.dropout_hidden(F.relu(self.fc3(x)))

        # output so no dropout here
        x = torch.sigmoid(self.fc4(x))

        return x