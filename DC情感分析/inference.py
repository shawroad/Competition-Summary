"""

@file  : inference.py

@author: xiaolu

@time  : 2020-06-02

"""
import torch
import json
from config import Config
from bert import Model
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm

def predict():
    # 准备一条数据
    test_data = pd.read_csv('./data/text_emotion/test_noLabel.csv')
    ids = test_data['ID'].tolist()
    test_txt = test_data['TXT'].tolist()
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')

    max_len = 500
    content = []
    for txt, id in tqdm(zip(test_txt, ids)):
        d = '[CLS]' + txt
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
        d = torch.LongTensor([d]).to(Config.device)
        id = torch.LongTensor(id).to(Config.device)
        mask = torch.LongTensor([mask]).to(Config.device)

        content.append((d, id, mask))

    # 加载模型
    model = Model(Config).to(Config.device)
    model.load_state_dict(torch.load('bert_sen_pretrain.bin', map_location='cpu'))
    print("模型加载成功...")
    model.eval()

    write_id = []
    write_pre_lab = []
    with torch.no_grad():
        for q, i in tqdm(zip(content, ids)):
            output = model(q)
            # print(output)   # tensor([-2.1185,  3.0377])
            pred_value = torch.argmax(output.data, 0).cpu().numpy()
            write_id.append(i)
            write_pre_lab.append(pred_value.tolist())
            # print(write_id)
            # print(write_pre_lab)
    # ID,Label
    file_path = 'submit_example.csv'
    name = ['ID']
    result = pd.DataFrame(columns=name, data=write_id)
    result['Label'] = write_pre_lab
    result.to_csv(file_path)



if __name__ == '__main__':
    predict()
