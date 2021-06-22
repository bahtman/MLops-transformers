import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)
@hydra.main(config_path="../../config", config_name='config.yaml')
def predict_model(config: DictConfig) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    accuracy_list = []
    model = torch.load(hparams['model_path'])

    X_test = pd.read_pickle(hparams['test_x_path'])
    y_test = pd.read_pickle(hparams['test_y_path'])
    X_test = torch.tensor(X_test)
    y_test = y_test.to_numpy()
    y_test = torch.tensor(y_test)
    testloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*(X_test, y_test)),
                batch_size=hparams['test_batch_size'], shuffle=True)

    model.eval()
    with torch.no_grad():
        tp=0; tn = 0; fp = 0; fn = 0
        for texts, labels in testloader:
            ps = model(texts)
            ps = (ps>0.5).float()
            #print("ps = ", ps)
            #top_p, top_class = ps.topk(1, dim=1)
            #print(top_class)
            #print("top_p = ", top_p, "and top class = ", top_class)
            equals = ps == labels.view(*ps.shape)
            for i in range(len(ps)):
                if ps[i] == 1:
                    if equals[i] == True:
                        tp += 1
                    if equals[i] == False:
                        fp += 1
                if ps[i] == 0:
                    if equals[i] == True:
                        tn += 1
                    if equals[i] == False:
                        fn += 1
            #print("equals = ", equals)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracy_list.append(accuracy.item()*100)
    batch_number = np.arange(len(accuracy_list))
    log.info(f"tp = {tp} fp =  {fp}  tn = {tn} fn = {fn}")
    log.info(f"mean of accuracy = {np.mean(accuracy_list)}")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    log.info(f"Precision {precision}")
    log.info(f"Recall {recall}")
    log.info(f"F1-score {2*(precision*recall)/(precision+recall)}")
    plt.figure()
    plt.plot(batch_number, accuracy_list)
    plt.legend(['Test set accuacy'])
    plt.xlabel('batch_number'), plt.ylabel('Accuacy')
    plt.show()
if __name__ == "__main__":
    predict_model()
