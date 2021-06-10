from transformers import DistilBertTokenizerFast
import pandas as pd

X_train = pd.read_pickle('../../data/processed/X_train.pkl')
X_test = pd.read_pickle('../../data/processed/X_test.pkl')
y_train = pd.read_pickle('../../data/processed/y_train.pkl')
y_test = pd.read_pickle('../../data/processed/y_test.pkl')


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


train_encodings = tokenizer(X_train, truncation=True, padding=True)
#val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer([X_test, truncation=True, padding=True)

import torch

class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SMSDataset(train_encodings, y_train)
#val_dataset = SMSDataset(val_encodings, val_labels)
test_dataset = SMSDataset(test_encodings, y_test)