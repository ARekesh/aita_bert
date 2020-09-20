# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:10:09 2020

@author: andrei
"""

from statistics import mean
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

print("Downloading and preparing dataframe...")
aita_data = pd.read_csv('aita_clean.csv')
aita_data_trimmed = aita_data[['body','is_asshole']].copy()

print("Dataframe size before dropping empty rows is: "+str(aita_data_trimmed.size))
aita_data_trimmed = aita_data_trimmed[aita_data_trimmed['body'].astype(str).map(len) > 50]
print("Dataframe size after dropping empty rows is: " +str(aita_data_trimmed.size))

aita_trimmed_texts = list(aita_data_trimmed['body'])
aita_trimmed_labels = list(aita_data_trimmed['is_asshole'])

train_texts, val_texts, train_labels, val_labels = train_test_split(aita_trimmed_texts, aita_trimmed_labels, test_size=.2)

#print(aita_data_train['body'].astype(str).apply(lambda x:len(x)).max())

print("Generating tokens...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
print("Tokens generated. Constructing dataset...")
class AITADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AITADataset(train_encodings, train_labels)
val_dataset = AITADataset(val_encodings, val_labels)

print("Dataset constructed. Initializing training...")
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

print("Training complete. Saving weights...")
model.save_pretrained("C:/Users/andrei/Documents/aita_models")
tokenizer.save_pretrained("C:/Users/andrei/Documents/aita_models")
print("Weights saved.")