"""

@file  : albert.py

@author: xiaolu

@time  : 2020-06-02

"""
from transformers import BertModel, BertConfig
from torch import nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./bert_pretrain/bert_config.json')
        self.bert = BertModel.from_pretrained('./bert_pretrain/pytorch_bert.bin', config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.gru = nn.GRU(input_size=config.hidden_size, hidden_size=256, num_layers=1, batch_first=True)

        self.fc = nn.Linear(768, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        step_output, pooled = self.bert(context, attention_mask=mask)
        # print(step_output.size())    # torch.Size([2, 10, 768])
        # print(pooled.size())   # torch.Size([2, 768])
        # exit()

        # output, last_output = self.gru(step_output)
        # print(last_output.size())
        # exit()

        # input: [batch, seq_len, input_size]
        # last_output: num_layers * num_directions, batch, hidden_size]

        # print(output.size())   # torch.Size([2, 10, 256])
        # print(last_output.size())   # torch.Size([1, 2, 256])

        # last_output = last_output.squeeze()

        out = self.fc(pooled)
        return out