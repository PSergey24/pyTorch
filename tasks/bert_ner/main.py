import torch
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
class BertModel(torch.nn.Module):
    def __init__(self, unique_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output


class DataSequence(Dataset):
    def __init__(self, df, tokenizer, labels_to_ids):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i), padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i, j, tokenizer, labels_to_ids) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


def align_label(texts, labels, tokenizer, labels_to_ids):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_all_tokens = True
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


def main():
    df = get_data()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    unique_labels, labels_to_ids, ids_to_labels = get_labels(df)

    df = df[0:100]
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    model = BertModel(unique_labels)
    train_loop(model, df_train, df_val, tokenizer, labels_to_ids)


def get_data():
    df = pd.read_csv('tasks/bert_ner/ner.csv')
    return df


def get_labels(df):
    labels = [i.split() for i in df['labels'].values.tolist()]

    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return unique_labels, labels_to_ids, ids_to_labels


def train_loop(model, df_train, df_val, tokenizer, labels_to_ids):
    LEARNING_RATE = 5e-3
    EPOCHS = 5
    BATCH_SIZE = 2

    train_dataset = DataSequence(df_train, tokenizer, labels_to_ids)
    val_dataset = DataSequence(df_val, tokenizer, labels_to_ids)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {val_loss: .3f} | Accuracy: {val_accuracy: .3f}')
