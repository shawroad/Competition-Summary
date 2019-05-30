"""

@file   : 蚂蚁金服语义相似性度量.py

@author : xiaolu

@time1  : 2019-05-30

"""
from keras import backend as K
from keras.layers import Embedding, LSTM, Bidirectional, Input
from keras.layers import Dense, Dropout, concatenate, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.replace('\n', '').split()
            sent1 = item[1].strip()
            sent2 = item[2].strip()
            label = item[-1]
            temp = [sent1, sent2, label]
            data.append(temp)
    return data


def build_char(data):
    # 我们基于字吧  所以这里整理字表
    word_list = []
    for item in data:
        sent = item[0] + item[1]
        word = list(sent)
        word_list.extend(word)
    # 去重，进行id的映射
    word_list = set(word_list)
    # word2id = {w: i for i, w in enumerate(word_list)}
    # word2id['UNK'] = 3000    # 不存在的字符标记为3000
    # id2word = {i: w for w, i in word2id.items()}
    # return word2id, id2word
    vocab = {j: i + 1 for i, j in enumerate(word_list)}
    vocab["UNK"] = 0
    return vocab

def process_data(data, word2id):
    # 将数据转为id序列 并将每条数据pad成同样的长度

    temp_data = []
    for item in data:
        sent1 = list(item[0])
        sent2 = list(item[1])
        # 对句子进行id的映射
        sent_1 = [word2id.get(c, 'UNK') for c in sent1]
        sent_2 = [word2id.get(c, 'UNK') for c in sent2]

        # 对句子1进行padding
        if len(sent_1) > 10:
            sent_1 = sent_1[:10]
        else:
            for i in range(10-len(sent_1)):
                sent_1.append(word2id.get('UNK'))

        # 对句子2进行padding
        if len(sent_2) > 10:
            sent_2 = sent_2[:10]
        else:
            for i in range(10-len(sent_2)):
                sent_2.append(word2id.get('UNK'))

        t = [sent_1, sent_2, item[2]]
        temp_data.append(t)
    return temp_data


def SiameseBiLSTM(vocab, max_length):

    K.clear_session()
    embedding = Embedding(len(vocab), 200, input_length=max_length)
    bilstm = Bidirectional(LSTM(128))

    # 对第一句话输入进行embedding
    sequence_input1 = Input(shape=(max_length,))
    embedded_sequences_1 = embedding(sequence_input1)
    x1 = bilstm(embedded_sequences_1)

    # 对第二句话输出进行embedding
    sequence_input2 = Input(shape=(max_length,))
    embedded_sequences_2 = embedding(sequence_input2)
    x2 = bilstm(embedded_sequences_2)

    # 将两次的输入进行拼接
    merged = concatenate([x1, x2])
    # 整体归一化
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(100, activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_input1, sequence_input2], outputs=preds)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == "__main__":
    path = './data/atec_nlp_sim_train_all.csv'
    data = load_data(path)
    word2id = build_char(data)
    # print(word2id)
    print(len(word2id))  # 2153
    # 将每句话转为id序列
    data_id = process_data(data, word2id)
    for i in data_id[:10]:
        print(i[0])
        print(i[1])
        print(i[2])

    Sens_1 = []
    Sens_2 = []
    labels = []
    for item in data_id:
        Sens_1.append(item[0])
        Sens_2.append(item[1])
        labels.append(item[2])
    maxlen = 10
    labels = np.array(labels)
    labels = labels.reshape((-1, 1))
    Sens_1 = np.array(Sens_1).reshape((-1, 10))
    Sens_2 = np.array(Sens_2).reshape((-1, 10))
    model = SiameseBiLSTM(word2id, maxlen)
    model.fit([Sens_1, Sens_2], labels, batch_size=32, epochs=2, validation_split=0.2)

    # 保存模型
    model.save('语义相似性度量.h5')



