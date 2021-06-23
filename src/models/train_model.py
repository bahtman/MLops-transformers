import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import wandb
import time
from model import MyAwesomeModel
from torch import optim
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)
@hydra.main(config_path="../../config", config_name='config.yaml')


def train_model(config: DictConfig) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])
    log.info(f'hparameters:  {hparams}')
    wandb.init(project="MLOps")
    device = torch.device("cuda" if hparams['cuda'] else "cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #num_workers = 2
    batch_size = hparams['batch_size']
    X_train = torch.tensor(pd.read_pickle(hparams['train_x_path']))
    y_train = torch.tensor(pd.read_pickle(hparams['train_y_path']).to_numpy())
    X_val = torch.tensor(pd.read_pickle(hparams['val_x_path']))
    y_val = torch.tensor(pd.read_pickle(hparams['val_y_path']).to_numpy())

    trainloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*(X_train, y_train)),
                batch_size=batch_size, shuffle=True)#, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*(X_val, y_val)),
                batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
    loss_list = []
    val_loss_list = []
    res = []

    log.info("Training day and night")
    model = MyAwesomeModel(hparams)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids = [0])

    wandb.watch(model, log_freq=500)
    criterion = torch.nn.BCELoss(reduction='none')
    #criterion = torch.nn.NLLLoss(torch.tensor([0.8]))
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    epochs = hparams['epochs']
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
            w = (hparams['loss_weight_1']*(1-labels)+hparams['loss_weight_2']*labels).detach()
            loss = (w*loss).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"train_loss": loss})
        else:
            loss_list.append(running_loss/len(trainloader))
            if e % 20 == 0:
                log.info(f"at epoch: {e} the Training loss is : {running_loss/len(trainloader)}") 
        with torch.no_grad():
            running_loss_val = 0
            model.eval()
            for texts, labels in valloader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(texts)
                output = torch.squeeze(output,1)
                loss_val = criterion(output.float(), labels.float())
                w_val = (hparams['loss_weight_1']*(1-labels)+hparams['loss_weight_2']*labels).detach()
                loss_val = (w_val*loss_val).mean()
                running_loss_val += loss_val.item()
                wandb.log({"val_loss": loss_val})
            else:
                val_loss_list.append(running_loss_val/len(trainloader))
                if e % 20 == 0:
                    log.info(f"at epoch: {e} the Validation loss is : {running_loss_val/len(valloader)}") 
        if (running_loss_val / len(valloader)) < lowest_val_loss:
            torch.save(model, hparams['model_path'])
            lowest_val_loss = running_loss_val/len(valloader)
        else:
            continue      
        #wandb.log({"texts": [wandb.texts(i) for i in texts]})
        end = time.time()
        res.append(end - start)
    res = np.array(res)
    log.info(f'Timing: {np.mean(res)} +- {np.std(res)}')
    #plt.figure()
    #epoch = np.arange(len(loss_list))
    #plt.plot(epoch, loss_list)
    #plt.plot(epoch, val_loss_list)
    #plt.legend(['Training loss and validation loss'])
    #plt.xlabel('Epochs'), plt.ylabel('Loss')
    #plt.show()
if __name__ == "__main__":
    train_model()