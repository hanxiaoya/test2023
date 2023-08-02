# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 词向量网络
        if config.embedding_pretrained is not None:
            # 使用不需要重新训练的、预训练好的词向量，加快训练速度、提升性能
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # 使用新定义的可训练的词词向量
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # LSTM网络
        # embed：词向量维度
        # hidden_size：隐藏层输出特征的长度（隐层神经元个数）
        # num_layers：隐藏层的层数
        # bidirectional：双向网络
        # batch_first： [batch_size, seq_len, embeding]
        # dropout：随机丢弃
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        # config.hidden_size * 2：双向LSTM的输出是隐层特征输出的2倍
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
