"""

@file  : data_utils.py

@author: xiaolu

@time  : 2019-10-09

"""
from config import Config


def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='utf8') as f:
        for line in f:
            x = line.strip() + "]"
            x = x.split(":")[1]
            if len(x) <= 5:
                continue
            if x[5] == '，':
                files_content += x

    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    print(len(erase))

    for key in erase:
        del counted_words[key]
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    words += (" ",)
    # word到id的映射
    word2id = dict((c, i) for i, c in enumerate(words))
    id2word = dict((i, c) for i, c in enumerate(words))
    word2id_fun = lambda x: word2id.get(x, len(words) - 1)
    print(len(words))

    return word2id_fun, id2word, words, files_content


if __name__ == '__main__':
    # 预处理数据
    word2id_fun, id2word, words, files_content = preprocess_file(Config)
    # print(words)
