"""

@file  : config.py

@author: xiaolu

@time  : 2020-06-02

"""
import torch


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    hidden_size = 768
    num_classes = 2

    num_epochs = 3  # 训练结果epoch
    learning_rate = 5e-5

    save_path = 'bert_sen_pretrain.bin'
    #
    # albert_config_path = './albert_pretrain/albert_config.json'
    # albert_model_path = './albert_pretrain/pytorch_albert.bin'

