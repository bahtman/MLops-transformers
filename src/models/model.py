from transformers import DistilBertTokenizerFast
from make_dataset import SpamDataSet
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


train_encodings = tokenizer(trainset, truncation=True, padding=True)
#val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(testset, truncation=True, padding=True)

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

train_dataset = SMSDataset(train_encodings, train_labels)
#val_dataset = SMSDataset(val_encodings, val_labels)
test_dataset = SMSDataset(test_encodings, test_labels)