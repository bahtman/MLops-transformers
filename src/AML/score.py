import json
import numpy as np
import os
import torch
from src.models.model import MyAwesomeModel
import string
import transformers
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import hydra

@hydra.main(config_path=".", config_name='hparms.yaml')

def init():
    global model, tokenizer, model_bert, stop_words
    hparams = config.experiment
    dir = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pth")
    model = MyAwesomeModel(hparams).load_state_dict(torch.load(dir))
    model.eval()
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased")
    model_bert = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def run(request):
        text = json.loads(request)
        tokenized = preprocess(text["sms"])
        # Run inference
        with torch.no_grad():
            out = torch.squeeze(model(tokenized),0).numpy()



        return {'output':f'The probability of the sms being spam is: {out[0]}'}


def preprocess(word):
    # remove punctuation
    s = word.translate(str.maketrans('', '', string.punctuation))
    #tokens = word_tokenize(s)
    tokens = s.split(" ")
    # removing stop-words
    cleaned_s = [w for w in tokens if w not in stop_words]
    first10 = " ".join(cleaned_s[:10])
    tokenized = tokenizer.encode(first10, add_special_tokens=True)
    input = torch.unsqueeze(torch.tensor(tokenized),0)
    with torch.no_grad():
        encoder_hidden_state = model_bert(input)
    cls = encoder_hidden_state[0][:, 0, :]
    return cls