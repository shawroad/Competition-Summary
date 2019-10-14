"""

@file  : config.py

@author: xiaolu

@time  : 2019-10-09

"""
class Config:
    poetry_file = 'dataset/poetry.txt'   # 数据集所在的位置
    weight_file = 'poetry_model.h5'   # 模型保存的位置
    # 根据前六个字预测第七个字
    max_len = 6
    batch_size = 32
    learning_rate = 0.001
