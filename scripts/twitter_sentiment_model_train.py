import pickle
import numpy as np
import pandas as pd
import sys
import re

from tqdm import tqdm

from transformers import LongformerConfig, LongformerTokenizerFast, LongformerForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import train_test_split

# Data Prep
data = pd.read_csv('../data/social_media/sentiment_twitter_dataset.csv', encoding='ISO-8859-1', names=['target', 'id', 'date', 'query', 'user', 'text'])

def preprocess(text):
    
    # remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    
    # remove links
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    return text

data.text = data.text.apply(preprocess)
data['text_length'] = data.text.apply(lambda x: len(x.split()))
text = data[data.text_length > 0]['text'].tolist()
labels = data[data.text_length > 0]['target'].tolist()


longformer_pretrained = 'allenai/longformer-base-4096'

tokenizer = LongformerTokenizerFast.from_pretrained(longformer_pretrained)

def encode_sent(x):
    encoding = tokenizer.encode_plus(x, max_length=70, padding='max_length', truncation=True, return_token_type_ids=False)
    return (encoding['input_ids'], encoding['attention_mask'])

token_ids = []
token_mask = []
for i in tqdm(range(len(text))):
    encoded = encode_sent(text[i])
    token_ids.append(encoded[0])
    token_mask.append(encoded[1])

label_map = {0: 0, 4: 1}
labels = [label_map[x] for x in labels]

train_ids, val_ids, train_mask, val_mask, train_label, val_label = train_test_split(
    token_ids,
    token_mask,
    labels,
    test_size=.2,
    random_state=42
)

train_dataset = TensorDataset(torch.tensor(train_ids), torch.tensor(train_mask), torch.tensor(train_label))
val_dataset = TensorDataset(torch.tensor(val_ids), torch.tensor(val_mask), torch.tensor(val_label))

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# Model Setup
config = LongformerConfig.from_pretrained(longformer_pretrained)
config.num_labels = 2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

model = LongformerForSequenceClassification.from_pretrained(longformer_pretrained, config=config)

model.cuda()

# Model Train

epochs = 20
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-6)
bias = [float(i) for i in '1,1'.split(',')]
weight = (1/torch.tensor(bias)).to(device)
softmax = torch.nn.Softmax(dim=1)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

loss = torch.nn.CrossEntropyLoss()

best_loss = np.inf
best_f1 = -np.inf

for i in range(1, epochs+1):
    with open('../results/current_results.txt', 'a') as f:
        f.write('\n--------------------------------------Epoch: {}---------------------------------------\n'.format(i))

    model.train()
    y_pred = []
    y_true = []
    y_preds = []
    batch_loss = []

    t = tqdm(iter(train_dataloader), leave=False, total=len(train_dataloader))

    for _, batch in enumerate(t):
        input_id = batch[0].to(device)
        mask_id = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_id, attention_mask=mask_id, labels=labels)

        loss_val = outputs.loss
        logits = outputs.logits

        # loss_val = loss(logits, labels)

        optimizer.zero_grad()

        loss_val.backward()

        preds = softmax(logits).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        y_true.extend(labels)
        y_pred.extend([np.argmax(pred) for pred in preds])
        y_preds.extend(preds)

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        optimizer.step()
        scheduler.step()

        batch_loss.append(loss_val.item())

    accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    auc = metrics.roc_auc_score(y_true, np.array(y_preds), multi_class='ovr')
    batch_loss = np.array(batch_loss).mean()

    
    with open('../results/current_results.txt', 'a') as f:
        f.write('Training Loss: {}\nTraining Accuracy: {}\nTraining F1: {}\nTraining AUC: {}'.format(batch_loss, accuracy, f1_score, auc))

    # validation
    if i % 5 == 0:
        model.eval()

        y_true_val = []
        y_pred_val = []
        y_preds = []
        val_loss = []

        
        with open('../results/current_results.txt', 'a') as f:
            f.write('\n----------------------------Validating Model------------------------\n')

        t = tqdm(iter(val_dataloader), leave=False, total=len(val_dataloader))

        for _, batch in enumerate(t):
            input_id = batch[0].to(device)
            mask_id = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_id, attention_mask=mask_id, labels=labels)

            loss_val = outputs.loss
            logits = outputs.logits

            preds = softmax(logits).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            y_true_val.extend(labels)
            y_pred_val.extend([np.argmax(pred) for pred in preds]) 
            y_preds.extend(preds)
            val_loss.append(loss_val.item())

        val_accuracy = metrics.accuracy_score(y_pred=y_pred_val, y_true=y_true_val)
        val_f1 = metrics.f1_score(y_pred=y_pred_val, y_true=y_true_val, average='weighted')
        val_auc = metrics.roc_auc_score(y_true_val, np.array(y_preds), multi_class='ovr')
        val_loss = np.array(val_loss).mean()

        
        with open('../results/current_results.txt', 'a') as f:
            f.write('Validation Loss: {}\nValidation Accuracy: {}\nValidation F1: {}\nValidation AUC: {}'.format(val_loss, val_accuracy, val_f1, val_auc))

        if best_f1 < val_f1:    
            with open('../results/current_results.txt', 'a') as f:
                f.write('New Best F1 saving model')
            model.save_pretrained('../models/LF_twitter_sentiment/')
            best_f1 = val_f1