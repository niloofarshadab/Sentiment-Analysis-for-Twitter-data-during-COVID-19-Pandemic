import numpy as np
import time
from os import listdir
from os.path import isfile, join
import torch
from transformers import BertForSequenceClassification, BertTokenizer
def get_sentiment(model, tokenizer, text_batch): #Large batches cause memory problems
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(encoding['input_ids']).logits
        pred = torch.argmax(outputs, axis=1)
    return pred.detach().numpy()

def get_sent(model, tokenizer, texts, batch_size=5):
    sent = ["NULL"] * len(texts)
    ind_buffer = []
    for i in range(len(texts)):
        if(texts[i] != "NULL"):
            ind_buffer.append(i)
        if(len(ind_buffer) == batch_size):
            sent_batch = get_sentiment(model, tokenizer, [texts[j] for j in ind_buffer])
            for j in range(len(sent_batch)):
                sent[ind_buffer[j]] = str(sent_batch[j])
            ind_buffer = list()
    if(len(ind_buffer) > 0):
        sent_batch = get_sentiment(model, tokenizer, [texts[j] for j in ind_buffer])
        for j in range(len(sent_batch)):
            sent[ind_buffer[j]] = str(sent_batch[j])
        ind_buffer = list()
    return sent


def rerun_file(file_path, new_path, model, tokenizer):

    fields, texts = [], []
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        i = 0
        while(line != ''):
            if(line[0] != '1'):
                line = f.readline().strip()
                continue
            splt = line.split('\t')
            text = splt[6].split('https://t.co')[0]
            if(len(text) < 10):
                line = f.readline().strip()
                continue
            i += 1

            fields.append(splt)
            texts.append(text)

            #id, date, time, language, geo, sentiment, text

            line = f.readline().strip()
    sents = get_sent(model, tokenizer, texts)
    print(file_path, new_path)
    with open(new_path, 'w+') as nf:
        for i in range(len(fields)):
            nf.write('\t'.join(fields[i][:5] + [sents[i]] + [texts[i]]) + '\n')

device = 'cpu'
model = BertForSequenceClassification.from_pretrained('./saves/sent_0.pt').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dir_path = "./data/day/"
file_paths = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f[:4] == 'sent']
k = 0
for fp in file_paths:
    k += 1
    if(isfile("./data/proc_day/proc" + fp[4:])):
        continue
    print(f'Rerunning file: {fp} - Progress: {k} / {len(file_paths)} ')
    rerun_file("./data/day/" + fp, "./data/proc_day/proc" + fp[4:], model, tokenizer)
