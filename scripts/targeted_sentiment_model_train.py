import pickle
import numpy as np
import sys

from tqdm import tqdm

from transformers import LongformerConfig, LongformerTokenizerFast, LongformerForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn import metrics


def batch_metrics(actual, pred):
    labels = []
    preds = []
    for i in range(actual.shape[0]):
        correct = np.where(actual[i] != 4)
        labels.extend(actual[i][correct])
        preds.extend(pred[i][correct, :][0].tolist())

    return (labels, preds)

# Data Prep
with open('../data/targeted/train.txt', 'rb') as f:
    train_data, train_labels = pickle.load(f)

with open('../data/targeted/val.txt', 'rb') as f:
    val_data, val_labels = pickle.load(f)

with open('../data/targeted/test.txt', 'rb') as f:
    test_data, test_labels = pickle.load(f)


longformer_pretrained = 'allenai/longformer-base-4096'

tokenizer = LongformerTokenizerFast.from_pretrained(longformer_pretrained)

def encode_sent(text, text_labels):
    tokenized_sentence = []
    labels = [0]

    for word, label in zip(text, text_labels):
        tw = tokenizer.tokenize(word)
        n_subwords = len(tw)

        labels.extend([label] * n_subwords)

    labels.extend([0])

    encoding = tokenizer.encode_plus(text, max_length=70, padding='max_length', truncation=True, return_token_type_ids=False)
    return (encoding['input_ids'], encoding['attention_mask'], labels)

segment_length = 70

train_ids = []
train_mask = []
train_label = []

for text, label in zip(train_data, train_labels):
    encoding = encode_sent(text, label)
    train_ids.append(encoding[0])
    train_mask.append(encoding[1])
    train_label.append(encoding[2])

for text, label in zip(val_data, val_labels):
    encoding = encode_sent(text, label)
    train_ids.append(encoding[0])
    train_mask.append(encoding[1])
    train_label.append(encoding[2])

val_ids = []
val_mask = []
val_label = []

for text, label in zip(test_data, test_labels):
    if label != 'neutral':
        encoding = encode_sent(text, label)
        val_ids.append(encoding[0])
        val_mask.append(encoding[1])
        val_label.append(encoding[2])

# pad labels to 70
train_label = pad_sequences(train_label, maxlen=segment_length, value=4, padding='post', dtype='long', truncating='post')
val_label = pad_sequences(val_label, maxlen=segment_length, value=4, padding='post', dtype='long', truncating='post')

train_dataset = TensorDataset(torch.tensor(train_ids), torch.tensor(train_mask), torch.tensor(train_label))
val_dataset = TensorDataset(torch.tensor(val_ids), torch.tensor(val_mask), torch.tensor(val_label))

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# Model Setup
config = LongformerConfig.from_pretrained(longformer_pretrained)
config.num_labels = 5

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

model = LongformerForTokenClassification.from_pretrained('../models/LF_targeted_sentiment/') # longformer_pretrained, config=config)

model.cuda()

# Model Train

epochs = 20
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-6)
bias = [float(i) for i in '1,1,1'.split(',')]
weight = (1/torch.tensor(bias)).to(device)
softmax = torch.nn.Softmax(dim=1)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

loss = torch.nn.CrossEntropyLoss()

best_loss = np.inf
best_f1 = 0.9020406754086824

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

        optimizer.zero_grad()

        loss_val.backward()

        preds = softmax(logits).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        labels, preds = batch_metrics(labels, preds)

        y_true.extend(labels)
        y_pred.extend([np.argmax(pred) for pred in preds])
        y_preds.extend(preds)

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        optimizer.step()
        scheduler.step()

        batch_loss.append(loss_val.item())

    batch_loss = np.array(batch_loss).mean()
    f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    
    with open('../results/current_results.txt', 'a') as f:
        f.write('Training Loss: {}\nF1 Score: {}'.format(batch_loss, f1_score))

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

            labels, preds = batch_metrics(labels, preds)

            y_true_val.extend(labels)
            y_pred_val.extend([np.argmax(pred) for pred in preds]) 
            y_preds.extend(preds)
            val_loss.append(loss_val.item())

        val_loss = np.array(val_loss).mean()
        val_f1 = metrics.f1_score(y_pred=y_pred_val, y_true=y_true_val, average='weighted')
        
        with open('../results/current_results.txt', 'a') as f:
            f.write('Validation Loss: {}\nValidation F1: {}'.format(val_loss, val_f1))

        if best_f1 < val_f1:    
            with open('../results/current_results.txt', 'a') as f:
                f.write('\nNew Best F1 saving model')
            model.save_pretrained('../models/LF_targeted_sentiment/')
            best_f1 = val_f1