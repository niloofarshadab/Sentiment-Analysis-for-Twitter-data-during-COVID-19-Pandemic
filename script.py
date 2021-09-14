import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import tweepy
import time
from os import listdir
from os.path import isfile, join

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Using device:", device)

def get_sentiment(model, tokenizer, text_batch): #Large batches cause memory problems
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(encoding['input_ids']).logits
        pred = torch.argmax(outputs, axis=1)
    return pred.detach().numpy()

def separate_dates():
    full_f = open("./data/full_dataset_clean.tsv")

    date_counts = {}

    header = full_f.readline().strip().split("\t")
    line = full_f.readline()
    i = 0
    while(line != ""):
        i += 1
        print(f"\r{i}/252342227",end="")
        vals = line.strip().split("\t")
        date = vals[1][:7]
        date_counts[date] = date_counts.get(date, 0) + 1
        with open(f"./data/month/{date}.tsv", "a+") as date_f:
            date_f.write(line)

        line = full_f.readline()
    print()
    print("\n".join([k + "\t" + str(date_counts[k]) for k in date_counts]))

def main2():
    model = BertForSequenceClassification.from_pretrained('./saves/sent_0.pt').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def main():
    model = BertForSequenceClassification.from_pretrained('./saves/sent_0.pt').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    file_exist = 0
    file_nonexist = 0

    twitter_limit = 100
    limit_time = 6 * 60
    month_dir = "./data/day/"
    month_files = [f for f in listdir(month_dir) if isfile(join(month_dir, f))]
    for f_path in month_files:
        old_file = join(month_dir, f_path)
        new_file = join(month_dir, "sent-" + f_path)

        if(f_path[-3:] != "tsv"):
            continue
        if(f_path[:4] == "sent"):
            continue
        if("sent-" + f_path in month_files):
            file_exist += 1
            continue
        file_nonexist += 1
        #continue

        for i in range(twitter_limit):
            print(f"\rRunning batches for {f_path} -  Progress: {i+1}/{twitter_limit}", end="")
            batch = read_next_batch(old_file, new_file)
            if(len(batch) == 0):
                continue
            tweet_ids = [x[0] for x in batch]
            responses = get_tweets(tweet_ids)
            texts = get_text(responses)
            #sent = get_sent(model, tokenizer, texts)
            #with open(new_file, "a+") as nf:
            #    for j in range(len(batch)):
            #        nf.write("\t".join(batch[j] + [sent[j]] + [texts[j]]) + '\n')
        print()
        for j in range(limit_time):
            print(f"\rLast Month: {f_path} - Sleeping: {j+1}/{limit_time}", end="")
            time.sleep(1)
        print()

    #print("Exists:", file_exist)
    #print("Doesn't", file_nonexist)

def read_next_batch(old_file, new_file, n=100):
    f = open(old_file, 'r')
    f2 = open(new_file, "a+")
    f2.close()
    f2 = open(new_file, "r")

    line_counter = 0
    l = f.readline()
    l2 = f2.readline()

    while(l != "" and l2 != ""):
        if(l.strip().split('\t')[3] != "en"):
            l = f.readline()
        else:
            line_counter += 1
            l = f.readline()
            l2 = f2.readline()

    buffer = []
    while(len(buffer) < 100 and l != ""):
        vals = l.strip().split('\t')
        if(len(vals) == 4):
            vals.append("NULL")
        if(vals[3] == "en"):
            buffer.append(vals)

        l = f.readline()
    return buffer


def get_tweets(ids):
    consumer_key = "---"
    consumer_secret = "----"
    bearer_token = "----"
    access_token = "-----"
    access_token_secret = "----"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    responses = api.statuses_lookup(ids, map=True, include_entities=False, trim_user=True)#, tweet_mode="extended")
    return responses

def get_text(responses):
    t = []
    for r in responses:
        if("text" in r.__dict__):
            #print(r.__dict__)

            #print("Full Text")
            #print(r.full_text)

            t.append(str(r.text.encode("utf-8").decode("ascii", "ignore")).replace('\t', '').replace('\n', ''))
        else:
            t.append("NULL")
    return t

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










def other(): #This runs everything. It would take ~400 days to run

    '''
    text = ["Have a fantastic day!", "Life is awful. I feel depressed."]
    pred = get_sentiment(model, tokenizer, text)

    v = {0: "negative", 1: "neutral", 2:"positive"}
    print([v[x] for x in pred])
    '''

    f = open("./data/full_dataset_clean.tsv")
    f2 = open("./data/with_sent.tsv", "a+")
    header = f.readline().strip().split("\t")

    line_counter = 0
    response_counter = 0
    english_counter = 0
    geo_counter = 0
    total_lines = 252342227
    l = f2.readline()
    while(l != ""):
        line_counter += 1
        f.readline()
        l = f2.readline()
        vals = line.strip().split("\t")
        if(vals[3] == "en"):
            english_counter += 1
        if(vals[4] != "NULL"):
            geo_counter += 1
        if(vals[5] != "NULL"):
            response_counter += 1
    start_line = line_counter
    start_time = time.perf_counter()

    line = f.readline()
    while(line):
        cur_time = time.perf_counter()
        estimated_time = ((total_lines - line_counter) * (cur_time - start_time) / (line_counter - start_line)) if start_line != line_counter else 0
        day_str = str(estimated_time // 86400)[:-2]
        time_str = day_str + time.strftime(':%H:%M:%S', time.gmtime(int(estimated_time % 86400)))

        print(f"\rPercent: {line_counter/total_lines:.3%} - Time: {time_str} - Line: {line_counter:,}/{total_lines:,} - English: {english_counter:,}/{line_counter:,} - Responses: {response_counter:,}/{line_counter:,} - Geo: {geo_counter:,}/{line_counter:,}", end="")
        vals = line.strip().split("\t")
        if(len(vals) == 4):
            vals.append("NULL")

        sent = "NULL"
        if(vals[3] == "en"):
            english_counter += 1
            try:
                response = api.get_status(vals[0], tweet_mode="extended")
                response_counter += 1
                sent = str(get_sentiment(model, tokenizer, [response.text])[0])
                if(vals[4] != "NULL"):
                    geo_counter += 1

            except tweepy.TweepError as e:
                print(e)
                exit()
                pass

        vals.append(sent)
        f2.write("\t".join(vals) + "\n")

        line = f.readline()
        line_counter += 1


if(__name__ == "__main__"):
    main()
