"""

@file  : poem_model.py

@author: xiaolu

@time  : 2019-10-09

"""
import random
import os
import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from config import Config
from data_utils import preprocess_file


class PoetryModel:
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = True
        self.config = config

        # 文件预处理 加载数据　并进行简单的预处理
        self.word2id_fun, self.id2word, self.words, self.files_content = preprocess_file(self.config)

        # 诗的list
        self.poems = self.files_content.split(']')  # 一条一条诗分开

        # 诗的总数量
        self.poems_num = len(self.poems)

        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.config.weight_file) and self.loaded_model:
            self.model = load_model(self.config.weight_file)
        else:
            self.train()

    def build_model(self):
        '''
        建立模型
        :return:
        '''
        print("building model")
        input_tensor = Input(shape=(self.config.max_len, len(self.words)))
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout)

        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def sample(self, preds, temperature=1.0):
        '''
        给一个输出热度
        :param preds: 预测的概率
        :param temperature: 越大 越保守　　越小　越open
        :return:
        '''
        # temperature 就是将概率进行放大 将概率大的权重调的更大　概率小的权重调的更小. 这样模型就比较open
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())

    def generate_sample_result(self, epoch, logs):
        '''
        训练过程中打印训练情况　每隔10个epoch打印一次预测
        :param epoch:
        :param logs:
        :return:
        '''
        if epoch % 10 != 0:
            return
        if not os.path.exists('out/out.txt'):
            os.makedirs('out/out.txt')
        # 打开文件 写入
        with open('out/out.txt', 'a', encoding='utf8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))

        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------Diversity {}--------------".format(diversity))
            generate = self.predict_random(temperature=diversity)
            print(generate)

            with open('out/out.txt', 'a', encoding='utf8') as f:
                f.write(generate + '\n')

    def predict_random(self, temperature):
        '''
        随机从库中选取一句开头的诗句,生成五言绝句
        :param temperature:
        :return:
        '''
        if not self.model:
            print("model not loaded")
            return

        index = random.randint(0, self.poems_num)  # 随机选一首诗
        sentence = self.poems[index][: self.config.max_len]  # 取出当前这首诗的前六个字
        generate = self.predict_sen(sentence, temperature=temperature)
        return generate

    def predict_sen(self, text, temperature=1):
        '''
        根据给出的前max_len个字，生成诗句
        :param text:
        :param temperature:
        :return:
        '''
        if not self.model:
            return

        max_len = self.config.max_len
        if len(text) < max_len:
            print("输入的长度不够,这里的长度必须是", max_len)
            return

        sentence = text[-max_len:]  # 每次取出倒数六个　就行下一个预测
        print("第一行:", sentence)
        generate = str(sentence)
        generate += self._preds(sentence, length=24 - max_len, temperature=temperature)
        return generate

    def _preds(self, sentence, length=23, temperature=1):
        '''
        :param sentence: 预测输入值
        :param length: 预测出的字符串长度
        :param temperature:
        :return:
        '''
        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, temperature)
            generate += pred
            sentence = sentence[1:] + pred  # 不要上一句的第一个字　加入新预测的这个字
        return generate

    def predict_first(self, char, temperature=1):
        '''
        根据给出的首个文字，生成五言绝句
        :param char:
        :param temperature:
        :return:
        '''
        if not self.model:
            print("model not loaded")
            return

        index = random.randint(0, self.poems_num)  # 随机选一首诗
        # 从选中的这首诗的模型截取五个字符再加入我们这个字符 相当于给个先验知识
        sentence = self.poems[index][1 - self.config.max_len:] + char
        generate = str(char)
        generate += self._preds(sentence, length=23, temperature=temperature)
        return generate

    def predict_hide(self, text, temperature=1):
        '''
        根据给4个字，生成藏头诗五言绝句
        :param text:
        :param temperature:
        :return:
        '''
        if not self.model:
            print("model not loaded")
            return
        if len(text) != 4:
            print("藏头诗的输入必须是4个字")
            return

        index = random.randint(0, self.poems_num)  # 随机选一首诗
        sentence = self.poems[index][(1 - self.config.max_len):] + text[0]   # 给定先验知识 然后把我们输入的第一个追加到后面
        generate = str(text[0])
        print('first line = ', sentence)

        # 相当于冷启动
        for i in range(5):  # 生成五个字
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(3):
            generate += text[i + 1]  # 第二个字加入
            sentence = sentence[1:] + text[i + 1]
            for i in range(5):
                next_char = self._pred(sentence, temperature)
                sentence = sentence[1:] + next_char
                generate += next_char
        return generate

    def _pred(self, sentence, temperature=1):
        '''
        内部使用方法，根据一串输入，返回单个预测字符
        :param sentence:
        :param temperature:
        :return:
        '''
        if len(sentence) < self.config.max_len:
            print("长度不够")
            return
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros((1, self.config.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2id_fun(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = self.id2word[next_index]
        return next_char

    def data_generator(self):
        '''
        数据生成器
        :return:
        '''
        i = 0
        while True:
            # 知道六个 预测第七个
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            # 相当于one_hot
            y_vec = np.zeros(shape=(1, len(self.words)), dtype=np.bool)
            y_vec[0, self.word2id_fun(y)] = 1.0

            x_vec = np.zeros(shape=(1, self.config.max_len, len(self.words)), dtype=np.bool)

            for t, char in enumerate(x):
                x_vec[0, t, self.word2id_fun(char)] = 1.0

            yield x_vec, y_vec
            i += 1

    def train(self):
        '''
        训练模型
        :return:
        '''
        print("training")
        number_of_epoch = len(self.files_content) - (self.config.max_len + 1) * self.poems_num
        number_of_epoch /= self.config.batch_size
        number_of_epoch = int(number_of_epoch / 1.5)
        print('epoches = ', number_of_epoch)
        print('poems_num = ', self.poems_num)
        print('len(self.files_content) = ', len(self.files_content))

        if not self.model:
            self.build_model()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


if __name__ == '__main__':
    # 实例化模型
    model = PoetryModel(Config)
    for i in range(3):
        # 藏头诗
        sen = model.predict_hide('思思漂亮')
        print(sen)

    exit()

    for i in range(3):
        # 给出第一句话进行预测
        sen = model.predict_sen('山为斜好几，')
        print(sen)

    for i in range(3):
        # 给出第一个字进行预测
        sen = model.predict_first('山')
        print(sen)

    for temp in [0.5, 1, 1.5]:
        # 随机抽取第一句话进行预测
        sen = model.predict_random(temperature=temp)
        print(sen)
