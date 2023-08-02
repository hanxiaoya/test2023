# coding: UTF-8
import os
import time
import torch
import numpy as np
from config import Config
from importlib import import_module
from train_eval import train, init_network
import argparse

def params():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char') # 分词器，默认按字切分
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 搜狗新闻:embedding_SougouNews.npz, 随机初始化:random
    args = params()
    embedding = args.embedding
    use_word = args.word
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif, build_embedding

    # 初始化参数配置
    config = Config(embedding, use_word, model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 构建3大数据集
    vocab, train_data, dev_data, test_data = build_dataset(config)
    # 更新词表大小
    config.n_vocab = len(vocab)
    if embedding == 'pre_trained' and config.pretrain_name!= '':   # 使用预训练词向量
        config.embed = build_embedding(vocab, config)
        config.embedding_pretrained = config.embedding_pretrained
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    # 创建模型实例
    x = import_module('models.' + model_name)
    model = x.Model(config).to(config.device)

    if model_name != 'Transformer':
        init_network(model)

    # # 显示网络参数
    # for name, w in model.named_parameters():
    #     print(name, w.shape)

    print("model.parameters：", model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    test(config, model, test_iter)