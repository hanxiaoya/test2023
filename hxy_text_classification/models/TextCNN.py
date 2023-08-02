# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # torch.nn.Conv2d(输入通道数, 卷积产生的通道数, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        # 经conv(x)卷积后输出结果：[batch_size, 卷积产生的通道数=num_filters, 卷积结果的高度=seq_len-k+1, 卷积结果的宽度=embedding]
        # squeeze(3):如果第4维上是1，则进行维度压缩，去掉第4维度；N=(W-F+2P)/S+1，经过计算第四维数据变成了1，所以就可以降维
        x = F.relu(conv(x)).squeeze(3)
        # F.max_pool1d(kernel_size,步长stride,填充padding,...)：
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # [batch_size, 卷积产生的通道数=num_filters]
        return x

    def forward(self, x):
        out = self.embedding(x[0]) # [batch_size, seq_len, embeding]=[128, 32, 300]
        # 第二维的前面增加一维，满足卷积的输入，此时的shape为[128, 1, 32, 300]
        out = out.unsqueeze(1)
        # 将三个向量拼接
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # [batch_size, num_filters * len(filter_sizes)]
        out = self.dropout(out)
        out = self.fc(out)
        return out