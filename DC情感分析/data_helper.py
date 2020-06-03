"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-06-02

"""
from transformers import BertTokenizer
import json
import pandas as pd
import numpy as np


def build_bert_input(data, label):
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    content = []
    for d, l in zip(data, labels):
        d = '[CLS]' + d
        d = tokenizer.tokenize(d)
        d = tokenizer.convert_tokens_to_ids(d)
        seq_len = len(d)
        if len(d) < max_len:
            mask = [1] * len(d) + [0] * (max_len - len(d))
            d += ([0] * (max_len - len(d)))
        else:
            mask = [1] * max_len
            d = d[:max_len]
            seq_len = max_len
        content.append((d, int(l), seq_len, mask))
    return content


if __name__ == '__main__':
    train_data_path = './data/text_emotion/train.csv'
    train_data = pd.read_csv(train_data_path)
    labels = train_data['Label'].tolist()
    text = train_data['txt'].tolist()
    # print(len(labels))
    # max_len = 0
    # tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    # length = []
    # for t in text:
    #     t = tokenizer.tokenize(t)
    #     t = tokenizer.convert_tokens_to_ids(t)
    #     length.append(len(t))
    # print(max(length))   # 3125
    # print(min(length))    # 11
    # print(np.mean(length))   # 311.87132
    max_len = 500
    data = build_bert_input(text, labels)
    json.dump(data, open('./data/train_data.json', 'w'))
