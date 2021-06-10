import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TrainingArguments

from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_metric


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
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments("test_trainer")



tokenized_train_data = tokenizer(train_data, batched=True)
tokenized_test_data = tokenizer(test_data, return_tensors='tf')
tokenized_train_label = tokenizer(train_label, return_tensors='tf')
tokenized_test_label = tokenizer(test_label, return_tensors='tf')

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_train_data, eval_dataset=tokenized_test_data
)
trainer.train()

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    compute_metrics=compute_metrics,
)
trainer.evaluate()