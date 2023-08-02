# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

def build_vocab(config):
    tokenizer = config.tokenizer
    max_size = config.MAX_VOCAB_SIZE
    min_freq = config.min_freq
    stopWord = readStopWord(config.stopWord_path) if os.path.exists(config.stopWord_path) else []
    vocab_dic = {}
    with open(config.build_vocab_data, 'r', encoding='UTF-8') as f:
        # 通过tqdm从单词表中读取一行单词，tqdm能够显示进度条
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            # subWords = [word for word in tokenizer(content) if word not in self.stopWordDict]  # 去掉停用词
            # wordCount = Counter(subWords)  # 统计词频
            for word in tokenizer(content):
                # 构建单词字典表
                if word not in stopWord: # 去停用词
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 对单词表进行排序
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 还原成字典
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 使用UNK,PAD填充单词表的尾部
        vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1})
        pkl.dump(vocab_dic, open(config.vocab_path, 'wb'))
        # unique_Label = list(set(labels))  # 标签去重
        # label2idx = dict(zip(unique_Label, list(range(len(unique_Label)))))  # {0: 0, 1: 1}
        # with open("./data/label2idx.json", "w", encoding="utf-8") as f:
        #     json.dump(label2idx, f)
    return vocab_dic

def readStopWord(stopWordPath):
    """
    读取停用词
    """
    with open(stopWordPath, "r") as f:
        stopWords = f.read()
        stopWordList = stopWords.splitlines()
        # 将停用词用列表的形式生成，之后查找停用词时会比较快
        # stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
    return stopWordList

def get_vocab(config):
    # load词表
    if os.path.exists(config.vocab_path):
        # 如果有现成的词表，则使用已有的词表（单词与索引的字典）
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config)
    return vocab

def load_dataset(path, vocab, config):
    contents = []
    pad_size = config.pad_size
    stopWord = readStopWord(config.stopWord_path) if os.path.exists(config.stopWord_path) else [] # 停用词
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            words_line = []
            # token = config.tokenizer(content)  # 分词
            token = [word for word in config.tokenizer(content) if word not in stopWord]  # 分词，去掉停用词
            seq_len = len(token)  # 句子长度
            if pad_size:   # 截断或填充PAD
                if len(token) < pad_size:
                    token.extend([config.PAD] * (pad_size - len(token)))  # 填充
                else:
                    token = token[:pad_size]    # 截断
                    seq_len = pad_size
            # word to id
            for word in token:  # 转换为词表对应的id
                words_line.append(vocab.get(word, vocab.get(config.UNK)))
            contents.append((words_line, int(label), seq_len))  # 句子id列表，标签id，句子长度
    return contents  # [([...], 0, 18), ([...], 1, 23), ...]

def build_dataset(config):
    # load词表
    vocab = get_vocab(config)
    print(f"Vocab size: {len(vocab)}")
    train = load_dataset(config.train_path, vocab, config)
    dev = load_dataset(config.dev_path, vocab, config)
    test = load_dataset(config.test_path, vocab, config)
    return vocab, train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, config):
        self.batches = batches  # 数据集
        self.batch_size = config.batch_size
        self.n_batches = len(batches) // config.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = config.device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_embedding_dim(config):
    emb_dim = config.embed
    f = open(config.pretrain_path, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:

            continue
        lin = line.strip().split(' ')
        if i == 1:
            emb_dim = len(lin[1:])
            break
    f.close()
    return emb_dim

def build_embedding(word_to_id, config):
    # pretrain_dir = "./dataset/data/sgns.sogou.char"
    # embedding_path = './dataset/data/embedding'
    # emb_dim = 300
    # word_to_id = pkl.load(open(vocab_dir, 'rb'))
    emb_dim = get_embedding_dim(config)
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    embeddings[len(word_to_id) - 1] = np.zeros(emb_dim)  # <PAD>
    f = open(config.pretrain_path, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(config.embedding_path, embeddings=embeddings)
    # np.savetxt(config.embedding_path, embeddings, delimiter=' ')  # 保存为txt
    return emb_dim

if __name__ == "__main__":
    '''提取预训练词向量'''
    embedding = 'pre_trained'
    use_word = False
    model_name = 'TextCNN'

    from config import Config
    config = Config(embedding, use_word, model_name)

    if os.path.exists(config.vocab_path):
        word_to_id = pkl.load(open(config.vocab_path, 'rb'))
    else:
        word_to_id = build_vocab(config.train_path, config.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(config.vocab_path, 'wb'))

    build_embedding(word_to_id, config)







