# coding: utf-8
import numpy as np
import torch
import os

class Config(object):
    """配置参数"""
    def __init__(self, embedding, use_word, model_name):
        self.model_name = model_name
        # 数据集路径
        self.dataset = 'dataset/data/'
        self.train_path = self.dataset + '/train.txt'                                # 训练集
        self.dev_path = self.dataset + '/dev.txt'                                    # 验证集
        self.test_path = self.dataset + '/test.txt'                                  # 测试集
        # 类别文件
        self.class_list = [x.strip() for x in open(
            self.dataset + '/class.txt', encoding='utf-8').readlines()]              # 类别名单
        # 单词表：是单词与其索引的对应表
        self.tokenizer = self.get_tokenizer(use_word)
        self.vocab_path = self.dataset + '/vocab.pkl'                                # 词表
        self.stopWord_name = ''
        # self.stopWord_name = 'english'                                                      # 停用词
        self.stopWord_path = self.dataset + self.stopWord_name if self.stopWord_name != '' else ''
        # 词向量表: 是索引与向量编码的对应表
        # self.pretrain_name = ''
        self.pretrain_name = 'sgns.sogou.char'
        self.pretrain_path = self.dataset + self.pretrain_name
        self.embedding_name = '{}.npz'.format(self.pretrain_name) if embedding == 'pre_trained' else embedding
        self.embedding_path = self.dataset + 'embedding_{}.npz'.format(self.pretrain_name) if embedding == 'pre_trained' else ''
        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding_path)["embeddings"].astype('float32'))\
            if embedding == 'pre_trained' and os.path.exists(self.embedding_path) else None  # 预训练词向量
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 100  # 字/词向量维度
        self.num_classes = len(self.class_list)  # 类别数
        self.build_vocab_data = self.train_path  # 如果没有现成的词表，则基于训练集，构建一个新的词表
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.MAX_VOCAB_SIZE = 10000  # 构建词表时，词表长度限制
        self.min_freq = 1    # 构建词表时，去掉词频小于该值的词
        self.UNK, self.PAD = '<UNK>', '<PAD>'  # 构建词表时，词表中的未知字，padding符号
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        # 训练数据保存
        self.save_path = self.dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = self.dataset + '/log/' + self.model_name
        # GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.model_params = self.model_params(self.model_name)  # 对应的模型参数

        self.func_dict = {'TextCNN': self.textcnn_params(), 'TextBiLSTM': self.textbilstm_params(), 'TextRCNN': self.textrcnn_params(),
                          'TextBiLSTM_Att': self.textBiLstm_att_params(), 'DPCNN': self.textdpcnn_params(), 'FastText': self.textfasttext_params(),
                          'Transformer': self.transformer()}
        self.model_params = self.func_dict[self.model_name]

    def get_tokenizer(self, use_word):
        # 指定分词器
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)，word-level
        else:
            # tokenizer：分词器，与英文不同，中文的单词是仅仅相邻的，中间没有空格，因此需要分词器进行分词。
            tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表, char-level
        return tokenizer

    def textcnn_params(self):
        # 模型参数
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        # 训练时的参数
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_epochs = 20  # epoch数
        self.num_epochs = 2  # epoch数
        # self.batch_size = 128  # mini-batch大小
        self.batch_size = 10  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率

    def textrcnn_params(self):
        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数

    def textbilstm_params(self, model_name='TextRNN'):
        # 模型参数
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        # 训练时的参数
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 2                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率

    def textBiLstm_att_params(self):
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.hidden_size2 = 64

    def textdpcnn_params(self):
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.num_filters = 250  # 卷积核数量(channels数)

    def textfasttext_params(self):
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.hidden_size = 256                                          # 隐藏层大小
        self.n_gram_vocab = 250499                                      # ngram 词表大小

    def transformer(self):
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 5e-4                                       # 学习率
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2




if __name__ == '__main__':
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'random'
    model_name = 'TextBiLSTM'
    print(Config(embedding, model_name).hidden_size)
